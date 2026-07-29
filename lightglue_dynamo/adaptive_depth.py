"""ONNX export and runtime support for LightGlue adaptive depth."""

from __future__ import annotations

import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from lightglue_dynamo.models import Pipeline

ADAPTIVE_DEPTH_METADATA_KEY = "lightglue_dynamo.adaptive_depth"
CONTROL_FLOW_DISABLED_OPTIMIZERS = frozenset({"MatMulAddFusion", "GatherSliceToSplitFusion"})


def is_adaptive_depth_model(model_path: Path) -> bool:
    """Detect an adaptive export without requiring ONNX at inference time."""
    markers = (ADAPTIVE_DEPTH_METADATA_KEY.encode(), b"LightGlueNaturalAdaptiveDepth")
    overlap = max(map(len, markers)) - 1
    previous = b""
    with model_path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            candidate = previous + chunk
            if any(marker in candidate for marker in markers):
                return True
            previous = candidate[-overlap:]
    return False


def _nest_natural_early_exits(output: Path, n_layers: int, minimum_depth: int = 1) -> None:
    """Rewrite export-friendly sequential If nodes into genuine nested exits."""
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(output)
    graph = model.graph
    conditional_nodes = [node for node in graph.node if node.op_type == "If"]
    expected_conditionals = n_layers - minimum_depth
    if len(conditional_nodes) != expected_conditionals:
        raise RuntimeError(f"expected {expected_conditionals} adaptive-depth If nodes, found {len(conditional_nodes)}")
    run_conditionals = conditional_nodes[:-1]
    final_conditional = conditional_nodes[-1]

    def branch(node: onnx.NodeProto, name: str) -> onnx.GraphProto:
        return next(attribute.g for attribute in node.attribute if attribute.name == name)

    def typed_value(name: str, template: onnx.ValueInfoProto) -> onnx.ValueInfoProto:
        value = copy.deepcopy(template)
        value.name = name
        return value

    def replace_capture(child: onnx.GraphProto, old_name: str, new_name: str) -> None:
        for node in child.node:
            for index, name in enumerate(node.input):
                if name == old_name:
                    node.input[index] = new_name

    def exit_graph(
        descriptor_name: str, depth: int, descriptor_type: onnx.ValueInfoProto, prefix: str
    ) -> onnx.GraphProto:
        descriptor_output = f"{prefix}_descriptor"
        depth_output = f"{prefix}_depth"
        return helper.make_graph(
            [
                helper.make_node("Identity", [descriptor_name], [descriptor_output], name=f"{prefix}_identity"),
                helper.make_node(
                    "Constant",
                    [],
                    [depth_output],
                    name=f"{prefix}_constant",
                    value=helper.make_tensor(f"{prefix}_value", TensorProto.INT64, [], [depth]),
                ),
            ],
            f"{prefix}_graph",
            [],
            [
                typed_value(descriptor_output, descriptor_type),
                helper.make_tensor_value_info(depth_output, TensorProto.INT64, []),
            ],
        )

    top_level_initializers = {initializer.name for initializer in graph.initializer}
    first_run_graph = branch(conditional_nodes[0], "then_branch")
    local_definitions = {output_name for node in first_run_graph.node for output_name in node.output if output_name}
    external_values = {
        input_name
        for node in first_run_graph.node
        for input_name in node.input
        if input_name and input_name not in local_definitions and input_name not in top_level_initializers
    }
    descriptor_candidates = external_values - {"repeat_interleave"}
    if len(descriptor_candidates) != 1:
        raise RuntimeError(f"could not identify first-layer descriptor capture: {sorted(descriptor_candidates)}")
    first_descriptor = descriptor_candidates.pop()

    def final_graph(current_descriptor: str) -> onnx.GraphProto:
        child = copy.deepcopy(branch(final_conditional, "then_branch"))
        old_descriptor = run_conditionals[-1].output[0] if run_conditionals else first_descriptor
        replace_capture(child, old_descriptor, current_descriptor)
        descriptor_output_info = copy.deepcopy(child.output[0])
        depth_output = f"natural_depth_{n_layers}"
        child.node.append(
            helper.make_node(
                "Constant",
                [],
                [depth_output],
                name=f"natural_depth_{n_layers}_constant",
                value=helper.make_tensor(f"natural_depth_{n_layers}_value", TensorProto.INT64, [], [n_layers]),
            )
        )
        del child.output[:]
        child.output.extend(
            [descriptor_output_info, helper.make_tensor_value_info(depth_output, TensorProto.INT64, [])]
        )
        child.name = f"natural_run_depth_{n_layers}"
        return child

    def run_graph(stage_index: int, current_descriptor: str) -> onnx.GraphProto:
        conditional = run_conditionals[stage_index]
        child = copy.deepcopy(branch(conditional, "then_branch"))
        old_descriptor = first_descriptor if stage_index == 0 else run_conditionals[stage_index - 1].output[0]
        replace_capture(child, old_descriptor, current_descriptor)

        descriptor_output_info = copy.deepcopy(child.output[0])
        stop_output = child.output[1].name
        descriptor_output = descriptor_output_info.name
        depth = minimum_depth + stage_index + 1
        continue_graph = (
            run_graph(stage_index + 1, descriptor_output)
            if stage_index + 1 < len(run_conditionals)
            else final_graph(descriptor_output)
        )
        stop_graph = exit_graph(descriptor_output, depth, descriptor_output_info, f"natural_exit_{depth}")
        nested_descriptor = f"natural_descriptor_{depth}_through_{n_layers}"
        nested_depth = f"natural_depth_{depth}_through_{n_layers}"
        child.node.append(
            helper.make_node(
                "If",
                [stop_output],
                [nested_descriptor, nested_depth],
                name=f"natural_if_stop_at_{depth}",
                then_branch=stop_graph,
                else_branch=continue_graph,
            )
        )
        del child.output[:]
        child.output.extend(
            [
                typed_value(nested_descriptor, descriptor_output_info),
                helper.make_tensor_value_info(nested_depth, TensorProto.INT64, []),
            ]
        )
        child.name = f"natural_run_depth_{depth}"
        return child

    top_producers = {output_name: node for node in graph.node for output_name in node.output}
    first_continue = conditional_nodes[0].input[0]
    continue_producer = top_producers[first_continue]
    if continue_producer.op_type != "Not" or len(continue_producer.input) != 1:
        raise RuntimeError("could not identify the first adaptive-depth stop predicate")
    first_stop = continue_producer.input[0]

    first_descriptor_info = copy.deepcopy(first_run_graph.output[0])
    first_descriptor_info.name = first_descriptor
    outer_stop_graph = exit_graph(
        first_descriptor, minimum_depth, first_descriptor_info, f"natural_exit_{minimum_depth}"
    )
    outer_continue_graph = run_graph(0, first_descriptor) if run_conditionals else final_graph(first_descriptor)
    final_descriptor_output = final_conditional.output[0]

    stop_producer_index = next(index for index, node in enumerate(graph.node) if first_stop in node.output)
    final_conditional_index = next(
        index for index, node in enumerate(graph.node) if node.name == final_conditional.name
    )
    removed_outputs = {
        output_name
        for node in graph.node[stop_producer_index + 1 : final_conditional_index + 1]
        for output_name in node.output
    }
    tail_inputs = {
        input_name for node in graph.node[final_conditional_index + 1 :] for input_name in node.input if input_name
    }
    replaced_tail_inputs = (removed_outputs & tail_inputs) - {final_descriptor_output}
    if len(replaced_tail_inputs) != 1:
        raise RuntimeError(f"could not identify the executed-depth value: {sorted(replaced_tail_inputs)}")
    executed_depth_output = replaced_tail_inputs.pop()

    nested_node = helper.make_node(
        "If",
        [first_stop],
        [final_descriptor_output, executed_depth_output],
        name=f"natural_if_stop_at_{minimum_depth}",
        then_branch=outer_stop_graph,
        else_branch=outer_continue_graph,
    )

    retained_nodes = [*graph.node[: stop_producer_index + 1], nested_node, *graph.node[final_conditional_index + 1 :]]
    del graph.node[:]
    graph.node.extend(retained_nodes)
    graph.name = "LightGlueNaturalAdaptiveDepth"
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, output, save_as_external_data=False)


def _prepare_control_flow_onnx_for_tensorrt(output: Path) -> None:
    """Normalize branch graphs for TensorRT's ONNX parser."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(output)

    def normalize_graph(graph: onnx.GraphProto) -> None:
        used_values = {name for node in graph.node for name in node.input}
        used_values.update(value.name for value in graph.output)
        for node in graph.node:
            if node.op_type == "LayerNormalization" and len(node.output) > 1:
                auxiliary_outputs = list(node.output[1:])
                if any(name in used_values for name in auxiliary_outputs):
                    raise RuntimeError(f"cannot remove used LayerNormalization outputs from {node.name}")
                del node.output[1:]
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    normalize_graph(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for child_graph in attribute.graphs:
                        normalize_graph(child_graph)

        producers = {output_name: node for node in graph.node for output_name in node.output}

        def constant_value(name: str) -> np.ndarray | None:
            node = producers.get(name)
            if node is None:
                return None
            if node.op_type == "Constant":
                value_attribute = next(
                    (attribute for attribute in node.attribute if attribute.name.startswith("value")), None
                )
                if value_attribute is None:
                    return None
                value = helper.get_attribute_value(value_attribute)
                if isinstance(value, onnx.TensorProto):
                    value = numpy_helper.to_array(value)
                return np.asarray(value)
            if node.op_type == "Reshape" and len(node.input) == 2:
                value = constant_value(node.input[0])
                shape = constant_value(node.input[1])
                if value is not None and shape is not None:
                    return value.reshape(tuple(int(dimension) for dimension in shape.flat))
            return None

        initializer_names = {initializer.name for initializer in graph.initializer}
        replaced_outputs: set[str] = set()
        for node in graph.node:
            if not node.op_type.startswith("Reduce") or len(node.input) < 2 or node.input[1] in initializer_names:
                continue
            axes_name = node.input[1]
            axes = constant_value(axes_name)
            if axes is None:
                continue
            graph.initializer.append(numpy_helper.from_array(axes.astype(np.int64), name=axes_name))
            initializer_names.add(axes_name)
            replaced_outputs.add(axes_name)

        if replaced_outputs:
            retained_nodes = [
                node for node in graph.node if not any(output_name in replaced_outputs for output_name in node.output)
            ]
            del graph.node[:]
            graph.node.extend(retained_nodes)

    normalize_graph(model.graph)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, output, save_as_external_data=False)


def export_adaptive_pipeline(
    *,
    pipeline: Pipeline,
    example_images: torch.Tensor,
    num_keypoints: int,
    output: Path,
    opset: int,
    dynamic_shapes: tuple[dict[int, object], ...] | None,
    custom_translation_table: dict[Any, Any],
) -> None:
    """Export extractor and adaptive matcher halves, then compose one ONNX model."""
    import onnx
    import torch

    from lightglue_dynamo.ops.shape_utils import shape_as_tensor

    class ExtractorPrefix(torch.nn.Module):
        def __init__(self, extractor: torch.nn.Module) -> None:
            super().__init__()
            self.extractor = extractor

        def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            shape = shape_as_tensor(images)
            height = shape[-2]
            width = shape[-1]
            extract_for_matching = getattr(self.extractor, "extract_for_matching", None)
            if callable(extract_for_matching):
                keypoints, descriptors = extract_for_matching(images)
            else:
                keypoints, _scores, descriptors, *_metadata = self.extractor(images)
            image_size = torch.stack([width, height]).to(device=keypoints.device, dtype=keypoints.dtype)
            if getattr(self.extractor, "normalize_by_long_edge", False):
                normalized_keypoints = (keypoints - image_size / 2) / (image_size.max() / 2)
            else:
                normalized_keypoints = 2 * keypoints / image_size - 1
            return keypoints, normalized_keypoints, descriptors

    extractor = ExtractorPrefix(pipeline.extractor).eval()
    matcher = pipeline.matcher.eval()
    if matcher.depth_confidence == -1:
        raise ValueError("adaptive export requires an enabled matcher depth_confidence")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".lightglue-adaptive-export-", dir=output.parent) as temporary:
        temporary_root = Path(temporary)
        extractor_path = temporary_root / "extractor.onnx"
        matcher_path = temporary_root / "matcher.onnx"
        staged_output = temporary_root / "adaptive.onnx"
        torch.onnx.export(
            extractor,
            (example_images,),
            str(extractor_path),
            input_names=["images"],
            output_names=["keypoints", "normalized_keypoints", "descriptors"],
            opset_version=opset,
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
            external_data=False,
            optimize=False,
            custom_translation_table=custom_translation_table,
        )
        torch.onnx.export(
            matcher,
            (
                example_images.new_zeros((2, num_keypoints, 2)),
                example_images.new_zeros((2, num_keypoints, matcher.input_dim)),
            ),
            str(matcher_path),
            input_names=["keypoints", "descriptors"],
            output_names=["matches", "mscores"],
            opset_version=opset,
            dynamo=True,
            external_data=False,
            optimize=False,
            custom_translation_table=custom_translation_table,
        )

        extractor_model = onnx.load(extractor_path)
        matcher_model = onnx.load(matcher_path)
        merged_model = onnx.compose.merge_models(
            extractor_model,
            matcher_model,
            io_map=[("normalized_keypoints", "keypoints"), ("descriptors", "descriptors")],
            # Prefix only the flat extractor. ONNX's prefix helper does not
            # rewrite lexical initializer references inside If subgraphs.
            prefix1="extractor/",
            name="LightGlueAdaptiveDepth",
        )

        renames = {"extractor/images": "images", "extractor/keypoints": "keypoints"}
        for node in merged_model.graph.node:
            for index, name in enumerate(node.input):
                node.input[index] = renames.get(name, name)
            for index, name in enumerate(node.output):
                node.output[index] = renames.get(name, name)
        for value in merged_model.graph.input:
            value.name = renames.get(value.name, value.name)
        for value in merged_model.graph.output:
            value.name = renames.get(value.name, value.name)

        metadata = {entry.key: entry.value for entry in merged_model.metadata_props}
        metadata[ADAPTIVE_DEPTH_METADATA_KEY] = str(matcher.depth_confidence)
        onnx.helper.set_model_props(merged_model, metadata)
        onnx.save(merged_model, staged_output, save_as_external_data=False)

        _nest_natural_early_exits(staged_output, matcher.n_layers)
        _prepare_control_flow_onnx_for_tensorrt(staged_output)
        onnx.checker.check_model(staged_output, full_check=True)
        staged_output.replace(output)
