import ast
from collections import defaultdict
import json
from math import log2

tfcoder_grammar = {
    'Tensor-Operations': ['tf.abs(x)', 'tf.add(x, y)', 'tf.add_n(inputs)', 'tf.argmax(input, axis)',
                        'tf.argmin(input, axis)', 'tf.argsort(values, axis, stable=True)',
                        'tf.argsort(values, axis, direction="DESCENDING", stable=True)',
                        'tf.boolean_mask(tensor, mask)', 'tf.broadcast_to(input, shape)', 'tf.cast(x, dtype)',
                        'tf.clip_by_value(t, clip_value_min, clip_value_max)', 'tf.concat(values, axis)',
                        'tf.constant(value)', 'tf.constant(value, dtype)', 'tf.divide(x, y)', 'tf.equal(x, y)',
                        'tf.exp(x)', 'tf.expand_dims(input, axis)', 'tf.eye(num_rows)', 'tf.eye(num_rows, num_columns)',
                        'tf.eye(num_rows, dtype)', 'tf.fill(dims, value)', 'tf.gather(params, indices)',
                        'tf.gather(params, indices, axis, batch_dims)', 'tf.gather_nd(params, indices)',
                        'tf.gather_nd(params, indices, batch_dims)', 'tf.greater(x, y)', 'tf.greater_equal(x, y)',
                        'tf.math.bincount(arr)', 'tf.math.ceil(x)', 'tf.math.count_nonzero(input)',
                        'tf.math.count_nonzero(input, axis)', 'tf.math.cumsum(x, axis)',
                        'tf.math.cumsum(x, axis, exclusive=True)', 'tf.math.divide_no_nan(x, y)',
                        'tf.math.floor(x)', 'tf.math.log(x)', 'tf.math.logical_and(x, y)',
                        'tf.math.logical_not(x)', 'tf.math.logical_or(x, y)', 'tf.math.logical_xor(x, y)',
                        'tf.math.negative(x)', 'tf.math.reciprocal(x)', 'tf.math.reciprocal_no_nan(x)',
                        'tf.math.segment_max(data, segment_ids)', 'tf.math.segment_mean(data, segment_ids)',
                        'tf.math.segment_min(data, segment_ids)', 'tf.math.segment_prod(data, segment_ids)',
                        'tf.math.segment_sum(data, segment_ids)', 'tf.math.squared_difference(x, y)',
                        'tf.math.top_k(input, k)', 'tf.math.unsorted_segment_max(data, segment_ids, num_segments)',
                        'tf.math.unsorted_segment_mean(data, segment_ids, num_segments)',
                        'tf.math.unsorted_segment_min(data, segment_ids, num_segments)',
                        'tf.math.unsorted_segment_prod(data, segment_ids, num_segments)',
                        'tf.math.unsorted_segment_sum(data, segment_ids, num_segments)', 'tf.matmul(a, b)',
                        'tf.maximum(x, y)', 'tf.minimum(x, y)', 'tf.multiply(x, y)', 'tf.not_equal(x, y)',
                        'tf.one_hot(indices, depth)', 'tf.ones(shape)', 'tf.ones_like(input)',
                        'tf.pad(tensor, paddings, mode="CONSTANT")', 'tf.pad(tensor, paddings, mode="CONSTANT", constant_values)',
                        'tf.pad(tensor, paddings, mode="REFLECT")', 'tf.pad(tensor, paddings, mode="SYMMETRIC")',
                        'tf.range(start)', 'tf.range(start, limit, delta)', 'tf.reduce_any(input_tensor, axis)',
                        'tf.reduce_all(input_tensor, axis)', 'tf.reduce_max(input_tensor)',
                        'tf.reduce_max(input_tensor, axis)', 'tf.reduce_mean(input_tensor)',
                        'tf.reduce_mean(input_tensor, axis)', 'tf.reduce_min(input_tensor)',
                        'tf.reduce_min(input_tensor, axis)', 'tf.reduce_prod(input_tensor, axis)',
                        'tf.reduce_sum(input_tensor)', 'tf.reduce_sum(input_tensor, axis)',
                        'tf.repeat(input, repeats)', 'tf.repeat(input, repeats, axis)',
                        'tf.reshape(tensor, shape)', 'tf.reverse(tensor, axis)', 'tf.roll(input, shift, axis)',
                        'tf.round(x)', 'tf.scatter_nd(indices, updates, shape)',
                        'tf.searchsorted(sorted_sequence, values, side="left")',
                        'tf.searchsorted(sorted_sequence, values, side="right")', 'tf.sequence_mask(lengths)',
                        'tf.sequence_mask(lengths, maxlen)', 'tf.shape(input)', 'tf.sign(x)',
                        'tf.sort(values, axis)', 'tf.sort(values, axis, direction="DESCENDING")', 'tf.sqrt(x)',
                        'tf.square(x)', 'tf.squeeze(input)', 'tf.squeeze(input, axis)', "tf.stack(values, axis)", "tf.subtract(x, y)",
                        "tf.tensor_scatter_nd_update(tensor, indices, updates)", "tf.tensordot(a, b, axes)", "tf.tile(input, multiples)",
                        "tf.transpose(a)", "tf.transpose(a, perm)", "tf.unique_with_counts(x)", "tf.unstack(value, axis)", "tf.where(condition)",
                        "tf.where(condition, x, y)", "tf.zeros(shape)", "tf.zeros_like(input)",
                        "tf.SparseTensor(indices, values, dense_shape)", "tf.sparse.add(a, b)", "tf.sparse.concat(axis, sp_inputs)",
                        "tf.sparse.expand_dims(sp_input, axis)", "tf.sparse.from_dense(tensor)", "tf.sparse.maximum(sp_a, sp_b)",
                        "tf.sparse.minimum(sp_a, sp_b)", "tf.sparse.reduce_max(sp_input, axis, output_is_sparse)", "tf.sparse.reduce_sum(sp_input, axis, output_is_sparse)",
                        "tf.sparse.reset_shape(sp_input)", "tf.sparse.reshape(sp_input, shape)", "tf.sparse.retain(sp_input, to_retain)",
                        "tf.sparse.slice(sp_input, start, size)", "tf.sparse.split(sp_input, num_split, axis)", "tf.sparse.to_dense(sp_input)",
                        "tf.sparse.to_dense(sp_input, default_value)", "tf.sparse.to_indicator(sp_input, vocab_size)", "tf.sparse.transpose(sp_input)", "tf.sparse.transpose(sp_input, perm)",
                        "IndexingAxis1Operation", "IndexingOperation", "PairCreationOperation",
                        "SingletonTupleCreationOperation", "SlicingAxis0BothOperation", "SlicingAxis0LeftOperation",
                        "SlicingAxis0RightOperation", "SlicingAxis1BothOperation", "SlicingAxis1LeftOperation",
                        "SlicingAxis1RightOperation", "TripleCreationOperation",
                        "PROVIDED_CONSTANT_WEIGHT", "COMMON_CONSTANT_WEIGHT", "PRIMITIVE_INPUT_AS_TENSOR_WEIGHT", "INPUT_VARIABLE_WEIGHT",
                        "AXIS_CONSTANT_WEIGHT", "SHAPE_CONSTANT_WEIGHT", "OUTPUT_SHAPE_TUPLE_WEIGHT", "OUTPUT_SHAPE_TUPLE_WEIGHT",
                        "CONSTANT_DTYPES_AND_WEIGHTS[tf.int32]", "CONSTANT_DTYPES_AND_WEIGHTS[tf.float32]", "CONSTANT_DTYPES_AND_WEIGHTS[tf.bool]",
                        "CONSTANT_DTYPES_AND_WEIGHTS[tf.int64]"]}

tfcoder_grammar_nonterminals = {
    'Tensor-Operations': ['tf.abs(x)', 'tf.add(x, y)', 'tf.add_n(inputs)', 'tf.argmax(input, axis)',
                        'tf.argmin(input, axis)', 'tf.argsort(values, axis, stable=True)',
                        'tf.argsort(values, axis, direction="DESCENDING", stable=True)',
                        'tf.boolean_mask(tensor, mask)', 'tf.broadcast_to(input, shape)', 'tf.cast(x, dtype)',
                        'tf.clip_by_value(t, clip_value_min, clip_value_max)', 'tf.concat(values, axis)',
                        'tf.constant(value)', 'tf.constant(value, dtype)', 'tf.divide(x, y)', 'tf.equal(x, y)',
                        'tf.exp(x)', 'tf.expand_dims(input, axis)', 'tf.eye(num_rows)', 'tf.eye(num_rows, num_columns)',
                        'tf.eye(num_rows, dtype)', 'tf.fill(dims, value)', 'tf.gather(params, indices)',
                        'tf.gather(params, indices, axis, batch_dims)', 'tf.gather_nd(params, indices)',
                        'tf.gather_nd(params, indices, batch_dims)', 'tf.greater(x, y)', 'tf.greater_equal(x, y)',
                        'tf.math.bincount(arr)', 'tf.math.ceil(x)', 'tf.math.count_nonzero(input)',
                        'tf.math.count_nonzero(input, axis)', 'tf.math.cumsum(x, axis)',
                        'tf.math.cumsum(x, axis, exclusive=True)', 'tf.math.divide_no_nan(x, y)',
                        'tf.math.floor(x)', 'tf.math.log(x)', 'tf.math.logical_and(x, y)',
                        'tf.math.logical_not(x)', 'tf.math.logical_or(x, y)', 'tf.math.logical_xor(x, y)',
                        'tf.math.negative(x)', 'tf.math.reciprocal(x)', 'tf.math.reciprocal_no_nan(x)',
                        'tf.math.segment_max(data, segment_ids)', 'tf.math.segment_mean(data, segment_ids)',
                        'tf.math.segment_min(data, segment_ids)', 'tf.math.segment_prod(data, segment_ids)',
                        'tf.math.segment_sum(data, segment_ids)', 'tf.math.squared_difference(x, y)',
                        'tf.math.top_k(input, k)', 'tf.math.unsorted_segment_max(data, segment_ids, num_segments)',
                        'tf.math.unsorted_segment_mean(data, segment_ids, num_segments)',
                        'tf.math.unsorted_segment_min(data, segment_ids, num_segments)',
                        'tf.math.unsorted_segment_prod(data, segment_ids, num_segments)',
                        'tf.math.unsorted_segment_sum(data, segment_ids, num_segments)', 'tf.matmul(a, b)',
                        'tf.maximum(x, y)', 'tf.minimum(x, y)', 'tf.multiply(x, y)', 'tf.not_equal(x, y)',
                        'tf.one_hot(indices, depth)', 'tf.ones(shape)', 'tf.ones_like(input)',
                        'tf.pad(tensor, paddings, mode="CONSTANT")', 'tf.pad(tensor, paddings, mode="CONSTANT", constant_values)',
                        'tf.pad(tensor, paddings, mode="REFLECT")', 'tf.pad(tensor, paddings, mode="SYMMETRIC")',
                        'tf.range(start)', 'tf.range(start, limit, delta)', 'tf.reduce_any(input_tensor, axis)',
                        'tf.reduce_all(input_tensor, axis)', 'tf.reduce_max(input_tensor)',
                        'tf.reduce_max(input_tensor, axis)', 'tf.reduce_mean(input_tensor)',
                        'tf.reduce_mean(input_tensor, axis)', 'tf.reduce_min(input_tensor)',
                        'tf.reduce_min(input_tensor, axis)', 'tf.reduce_prod(input_tensor, axis)',
                        'tf.reduce_sum(input_tensor)', 'tf.reduce_sum(input_tensor, axis)',
                        'tf.repeat(input, repeats)', 'tf.repeat(input, repeats, axis)',
                        'tf.reshape(tensor, shape)', 'tf.reverse(tensor, axis)', 'tf.roll(input, shift, axis)',
                        'tf.round(x)', 'tf.scatter_nd(indices, updates, shape)',
                        'tf.searchsorted(sorted_sequence, values, side="left")',
                        'tf.searchsorted(sorted_sequence, values, side="right")', 'tf.sequence_mask(lengths)',
                        'tf.sequence_mask(lengths, maxlen)', 'tf.shape(input)', 'tf.sign(x)',
                        'tf.sort(values, axis)', 'tf.sort(values, axis, direction="DESCENDING")', 'tf.sqrt(x)',
                        'tf.square(x)', 'tf.squeeze(input)', 'tf.squeeze(input, axis)', "tf.stack(values, axis)", "tf.subtract(x, y)",
                        "tf.tensor_scatter_nd_update(tensor, indices, updates)", "tf.tensordot(a, b, axes)", "tf.tile(input, multiples)",
                        "tf.transpose(a)", "tf.transpose(a, perm)", "tf.unique_with_counts(x)", "tf.unstack(value, axis)", "tf.where(condition)",
                        "tf.where(condition, x, y)", "tf.zeros(shape)", "tf.zeros_like(input)",
                        "tf.SparseTensor(indices, values, dense_shape)", "tf.sparse.add(a, b)", "tf.sparse.concat(axis, sp_inputs)",
                        "tf.sparse.expand_dims(sp_input, axis)", "tf.sparse.from_dense(tensor)", "tf.sparse.maximum(sp_a, sp_b)",
                        "tf.sparse.minimum(sp_a, sp_b)", "tf.sparse.reduce_max(sp_input, axis, output_is_sparse)", "tf.sparse.reduce_sum(sp_input, axis, output_is_sparse)",
                        "tf.sparse.reset_shape(sp_input)", "tf.sparse.reshape(sp_input, shape)", "tf.sparse.retain(sp_input, to_retain)",
                        "tf.sparse.slice(sp_input, start, size)", "tf.sparse.split(sp_input, num_split, axis)", "tf.sparse.to_dense(sp_input)",
                        "tf.sparse.to_dense(sp_input, default_value)", "tf.sparse.to_indicator(sp_input, vocab_size)", "tf.sparse.transpose(sp_input)", "tf.sparse.transpose(sp_input, perm)",
                        "IndexingAxis1Operation", "IndexingOperation", "PairCreationOperation",
                        "SingletonTupleCreationOperation", "SlicingAxis0BothOperation", "SlicingAxis0LeftOperation",
                        "SlicingAxis0RightOperation", "SlicingAxis1BothOperation", "SlicingAxis1LeftOperation",
                        "SlicingAxis1RightOperation", "TripleCreationOperation"],
    'Constant-Operations': ["PROVIDED_CONSTANT_WEIGHT", "COMMON_CONSTANT_WEIGHT", "PRIMITIVE_INPUT_AS_TENSOR_WEIGHT", 
                        "AXIS_CONSTANT_WEIGHT", "SHAPE_CONSTANT_WEIGHT", "OUTPUT_SHAPE_TUPLE_WEIGHT", "INPUT_VARIABLE_WEIGHT",
                        "CONSTANT_DTYPES_AND_WEIGHTS[tf.int32]", "CONSTANT_DTYPES_AND_WEIGHTS[tf.float32]", 
                        "CONSTANT_DTYPES_AND_WEIGHTS[tf.bool]",
                        "CONSTANT_DTYPES_AND_WEIGHTS[tf.int64]"]}

class PatternExtractor(ast.NodeVisitor):
    def __init__(self):
        self.pattern_counts = defaultdict(int)

    def visit_Subscript(self, node):
        if isinstance(node.slice, ast.Tuple):
            elements = node.slice.elts
            if len(elements) == 2:
                first_dim, second_dim = elements
                if isinstance(first_dim, ast.Slice) and (first_dim.lower is None and first_dim.upper is None):
                    if not isinstance(second_dim, ast.Slice):
                        self.pattern_counts['IndexingAxis1Operation'] += 1
            for dim, item in enumerate(node.slice.elts):
                if isinstance(item, ast.Slice):
                    if item.lower is None and item.upper is None:
                        pass
                    elif item.lower is not None and item.upper is None:
                        if dim == 1:
                            self.pattern_counts['SlicingAxis1LeftOperation'] += 1
                    elif item.lower is None and item.upper is not None:
                        if dim == 1:
                            self.pattern_counts['SlicingAxis1RightOperation'] += 1
                    elif item.lower is not None and item.upper is not None:
                        if dim == 1:
                            self.pattern_counts['SlicingAxis1BothOperation'] += 1
        elif isinstance(node.slice, ast.Slice):
            if node.slice.lower and node.slice.upper:
                self.pattern_counts['SlicingAxis0BothOperation'] += 1
            elif node.slice.lower:
                self.pattern_counts['SlicingAxis0LeftOperation'] += 1
            elif node.slice.upper:
                self.pattern_counts['SlicingAxis0RightOperation'] += 1

        else:
            self.pattern_counts['IndexingOperation'] += 1
        self.generic_visit(node)

    def visit_Tuple(self, node):
        # Handle tuple creation operations
        num_elements = len(node.elts)
        if num_elements == 1:
            self.pattern_counts['SingletonTupleCreationOperation'] += 1
        elif num_elements == 2:
            self.pattern_counts['PairCreationOperation'] += 1
        elif num_elements == 3:
            self.pattern_counts['TripleCreationOperation'] += 1

        self.generic_visit(node)

class TensorFlowOperationExtractor(ast.NodeVisitor):
    def __init__(self):
        self.tf_operation_counts = {
            op: 0 for op in tfcoder_grammar['Tensor-Operations']}

    def visit_Call(self, node):
        operation = self.get_full_function_name(node.func)
        #if isinstance(node.func, ast.Attribute) and hasattr(node.func.value, 'id') and node.func.value.id == 'tf':
        if operation.startswith('tf.'):
            arg_signature = self.get_arg_signature(node)
            operation_dict = self.tf_operation_counts
            matching_keys = [key for key in operation_dict.keys(
            ) if key.startswith(operation + '(')]
            if len(matching_keys) > 1:
                num_args_in_call = len(arg_signature)
                matches_by_arg_count = []
                for key in matching_keys:
                    arg_part = key[len(operation)+1:-1]
                    num_args_in_key = len(
                        arg_part.split(', ')) if arg_part else 0
                    if num_args_in_key == num_args_in_call:
                        matches_by_arg_count.append(key)
                if len(matches_by_arg_count) == 1:
                    matched_by_arg_count = matches_by_arg_count[0]
                    operation_dict[matched_by_arg_count] += 1
                else:
                    if 'tf.pad' in operation:
                        if '"CONSTANT"' in arg_signature:
                            operation_dict['tf.pad(tensor, paddings, mode="CONSTANT")'] += 1
                    elif 'tf.searchsorted' in operation:
                        if 'left' in arg_signature:
                            operation_dict['tf.searchsorted(sorted_sequence, values, side="left")'] += 1
                        elif 'right' in arg_signature:
                            operation_dict['tf.searchsorted(sorted_sequence, values, side="right")'] += 1
                    elif 'tf.eye' in operation:
                        if 'dtype' in arg_signature:
                            operation_dict['tf.eye(num_rows, dtype)'] += 1
                        else:
                            operation_dict['tf.eye(num_rows, num_columns)'] += 1
            elif len(matching_keys) == 1:
                operation_dict[matching_keys[0]] += 1
        self.generic_visit(node)

    def get_full_function_name(self, node):
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        parts.reverse()
        return '.'.join(parts)
    
    def get_arg_signature(self, node):
        signature = []
        for arg in node.args:
            if isinstance(arg, ast.Str):
                signature.append(f'"{arg.s}"')
            elif isinstance(arg, ast.Name):
                signature.append(arg.id)
            else:
                signature.append('arg')
        for kw in node.keywords:
            kw_value = f'"{kw.value.s}"' if isinstance(
                kw.value, ast.Str) else 'value'
            signature.append(f"{kw.arg}")
            signature.append(f"{kw_value}")
        return signature

def extract_operations(source_code):
    ast_tree = ast.parse(source_code)
    extractor = TensorFlowOperationExtractor()
    pattern_extractor = PatternExtractor()
    extractor.visit(ast_tree)
    pattern_extractor.visit(ast_tree)
    return extractor.tf_operation_counts, pattern_extractor.pattern_counts

def laplace_smoothing(grammar, tf_operation_counts, param, alpha=1):
    smoothed_probabilities = {
        param: {},
    }
    total_tf_operations = sum(tf_operation_counts.values(
    )) + (len(grammar[param]) * alpha)
    for operation in grammar[param]:
        count = tf_operation_counts.get(operation, 0) + alpha
        smoothed_probabilities[param][operation] = count / \
            total_tf_operations
    return smoothed_probabilities

alpha = 1
def compute_costs(smoothed_probabilities, param):
    costs = {
        param: {}
    }
    for operation_type in smoothed_probabilities:
        for operation, probability in smoothed_probabilities[operation_type].items():
            costs[operation_type][operation] = round(-log2(probability))
    return costs

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def process_tasks(tasks):
    for task in tasks:
        task_id = task['name']
        print("task_id:", task_id)
        completions = task['normalized_completions'] # task['response']['completions']
        #print("aggregate_constant_count", task['response']['aggregate_constant_count'])
        aggregate_constant_count = task['aggregate_constant_count']
        aggregate_tf_operation_counts = defaultdict(int)
        for completion in completions:
            try:
                tf_operation_counts, patterns = extract_operations(completion)
                for operation, count in tf_operation_counts.items():
                    aggregate_tf_operation_counts[operation] += count

                for op, c in patterns.items():
                    aggregate_tf_operation_counts[op] += c
            except Exception as e:
                print(completion)
                print(f"Error processing completion for task {task_id}: {e}")
                continue
        
        # updating the constant operations
        for key, value in aggregate_constant_count.items():
            if key == "provided":
                aggregate_tf_operation_counts["PROVIDED_CONSTANT_WEIGHT"] += value
            elif key == "common":
                aggregate_tf_operation_counts["COMMON_CONSTANT_WEIGHT"] += value
            elif key == "input_var":
                aggregate_tf_operation_counts["INPUT_VARIABLE_WEIGHT"] += value
            elif key == "axis":
                aggregate_tf_operation_counts["AXIS_CONSTANT_WEIGHT"] += value
            elif key == "shape":
                aggregate_tf_operation_counts["SHAPE_CONSTANT_WEIGHT"] += value
            elif key == "shape_tuple":
                aggregate_tf_operation_counts["OUTPUT_SHAPE_TUPLE_WEIGHT"] += value
            elif key == "tf_int32":
                aggregate_tf_operation_counts["CONSTANT_DTYPES_AND_WEIGHTS[tf.int32]"] += value
            elif key == "tf_float32":
                aggregate_tf_operation_counts["CONSTANT_DTYPES_AND_WEIGHTS[tf.float32]"] += value
            elif key == "tf_bool":
                aggregate_tf_operation_counts["CONSTANT_DTYPES_AND_WEIGHTS[tf.bool]"] += value
            elif key == "tf_int64":
                aggregate_tf_operation_counts["CONSTANT_DTYPES_AND_WEIGHTS[tf.int64]"] += value
            else:
                aggregate_tf_operation_counts["CONSTANT_DTYPES_AND_WEIGHTS[tf.int64]"] += 0 # todo
        #print("aggregate_tf_operation_counts:", aggregate_tf_operation_counts)
        # PCFG1
        smoothed_probabilities = laplace_smoothing(
            tfcoder_grammar, aggregate_tf_operation_counts, 'Tensor-Operations', alpha=1)
        task["smoothed_probabilities"] = smoothed_probabilities
        task["costs"] = compute_costs(smoothed_probabilities, 'Tensor-Operations')
        # PCFG2 
        smoothed_probabilities_t = laplace_smoothing(
            tfcoder_grammar_nonterminals, aggregate_tf_operation_counts, 'Tensor-Operations', alpha=1)
        smoothed_probabilities_c = laplace_smoothing(
            tfcoder_grammar_nonterminals, aggregate_tf_operation_counts, 'Constant-Operations', alpha=1)
        costs_t = compute_costs(smoothed_probabilities_t, 'Tensor-Operations')
        costs_c = compute_costs(smoothed_probabilities_c, 'Constant-Operations')
        for operation, cost in costs_c['Constant-Operations'].items():
            costs_t['Tensor-Operations'][operation] = cost
        #task["costs"] = costs_t
    return tasks

def main():
    tasks = read_json_file("tfcoder_output.deepseek-ai__deepseek-coder-33b-instruct.json")
    new_tasks = process_tasks(tasks)
    with open("tfcoder_gpt3_100.json", 'w') as json_file:
        json.dump(new_tasks, json_file, indent=4)

if __name__ == "__main__":
    main()