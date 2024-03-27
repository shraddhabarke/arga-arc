from arga_ast_generator import *
from arga_ast_generator import _Ast
from collections import defaultdict
from lark import Tree, Token
from task import *
from VocabMaker import *
from filters import *
import enum

with open("dsl/v0_3/dsl.lark", "r") as f:
    arga_dsl_grammar = f.read()

parser = dsl_parser.Parser.new()
xformer = ast_utils.create_transformer(this_module, ToAst())
gens_dir = "dsl/v0_3/generations/20240318T231857_gpt-4-0125-preview_30"

# get the AST program from the generations directory
def test_file(filename, parser, xformer):
    with open(filename, "r") as f:
        lib = "(" + f.read() + ")"
    print(f"Testing {filename}...")
    t = parser.lib_parse_tree(lib)
    ast = xformer.transform(t)
    return ast

Color._sizes = {color.name: 1 for color in Color}
Dir._sizes = {dir.name: 1 for dir in Dir}
Overlap._sizes = {overlap.name: 1 for overlap in Overlap}
Rotation_Angle._sizes = {overlap.name: 1 for overlap in Rotation_Angle}
Symmetry_Axis._sizes = {overlap.name: 1 for overlap in Symmetry_Axis}
RelativePosition._sizes = {relativepos.name: 1 for relativepos in RelativePosition}
ImagePoints._sizes = {imagepts.name: 1 for imagepts in ImagePoints}

FColor._sizes = {color.name: 1 for color in FColor}
Shape._sizes = {shape.name: 1 for shape in Shape}
Degree._sizes = {degree.value: 1 for degree in Degree._all_values}
Size._sizes = {s.value: 1 for s in Size._all_values}
Column._sizes = {col.value: 1 for col in Column._all_values}
Row._sizes = {row.value: 1 for row in Row._all_values}
Height._sizes = {height.value: 1 for height in Height._all_values}
Width._sizes = {width.value: 1 for width in Width._all_values}

object_ids, size_values, degree_values, height_values, width_values, column_values, row_values = [], [], [], [], [], [], []
def processtask(taskNumber):
    task = Task("ARC/data/training/" + taskNumber + ".json")
    task.abstraction = "nbccg"
    task.input_abstracted_graphs_original[task.abstraction] = [getattr(input, Image.abstraction_ops[task.abstraction])() for
                                                        input in task.train_input]
    task.output_abstracted_graphs_original[task.abstraction] = [getattr(output, Image.abstraction_ops[task.abstraction])() for
                                                        output in task.train_output]
    task.get_static_inserted_objects()
    task.get_static_object_attributes(task.abstraction)
    setup_size_and_degree_based_on_task(task)
    setup_objectids(task)
    vocabMakers = [ObjectId, FColor, Degree, Height, Width, Size, Shape, Row, Column, IsDirectNeighbor, IsDiagonalNeighbor, 
                IsAnyNeighbor, FilterByColor, FilterBySize, FilterByDegree, FilterByShape, FilterByHeight,
                FilterByColumns, FilterByNeighborColor, FilterByNeighborSize, FilterByNeighborDegree, Not, 
                And, Or, VarAnd]
    vocab = VocabFactory.create(vocabMakers)
    for leaf in list(vocab.leaves()):
        if isinstance(leaf, SizeValue) and not isinstance(leaf, enum.Enum):
            size_values.append(str(leaf.value))
        elif isinstance(leaf, DegreeValue) and not isinstance(leaf, enum.Enum):
            degree_values.append(str(leaf.value))
        elif isinstance(leaf, HeightValue) and not isinstance(leaf, enum.Enum):
            height_values.append(str(leaf.value))
        elif isinstance(leaf, WidthValue) and not isinstance(leaf, enum.Enum):
            width_values.append(str(leaf.value))
        elif isinstance(leaf, ColumnValue) and not isinstance(leaf, enum.Enum):
            column_values.append(str(leaf.value))
        elif isinstance(leaf, RowValue) and not isinstance(leaf, enum.Enum):
            row_values.append(str(leaf.value))
        elif isinstance(leaf, ObjectIdValue) and not isinstance(leaf, enum.Enum):
            object_ids.append(str(leaf.value))
    file_path = f"output.txt"
    ast_program = test_file(file_path, parser, xformer)
    print("print:", ast_program)
    print("object-ids:", object_ids)
    return ast_program

##--------------------- Computing transform probabilities ---------------------------------------------------------
transform_operations = {'UpdateColor', 'HollowRectangle', 'FillRectangle', 'AddBorder', 'MoveNode', 
                        'ExtendNode', 'MoveNodeMax', 'Mirror', 'Flip', 'RotateNode', 'NoOp', 'Insert', 'Transforms'}

# Transform grammar rules
transform_rules = {
    'Transform': [
        'NoOp',
        'UpdateColor',
        'MoveNode',
        'ExtendNode',
        'MoveNodeMax',
        'RotateNode',
        'AddBorder',
        'FillRectangle',
        'HollowRectangle',
        'Mirror',
        'Flip',
        'Insert',
        'Transforms'
    ],
    'Color': [
        'O', 'B', 'R', 'G', 'Y', 'X', 'F', 'A', 'C', 'W', 'most', 'least'
    ],
    'Direction': ['U', 'D', 'L', 'R', 'UL', 'UR', 'DL', 'DR'],
    'Variable': ['Var'],
    'Overlap': ['True', 'False'],
    'Rotation_Angle': ['90', '180', '270'],
    'Symmetry_Axis': ['VERTICAL', 'HORIZONTAL', 'DIAGONAL_LEFT', 'DIAGONAL_RIGHT'],
    'ImagePoints': ['TOP', 'BOTTOM', 'LEFT', 'RIGHT', 'TOP_LEFT', 'TOP_RIGHT', 'BOTTOM_LEFT', 'BOTTOM_RIGHT'],
    'RelativePosition': ['SOURCE', 'TARGET', 'MIDDLE'],
    'ObjectId': object_ids
}

# Step 1: PCFG initialization and the extraction of counts from program ASTs
def initialize_uniform_pcfg(grammar_rules):
    pcfg = defaultdict(dict)
    for non_terminal, productions in grammar_rules.items():
        uniform_prob = 1.0 / len(productions)
        for production in productions:
            pcfg[non_terminal][production] = uniform_prob
    return pcfg

init_transform_pcfg = initialize_uniform_pcfg(transform_rules)

def increment_transforms(program, init_transform_pcfg):
    """
    Increment the 'Transforms' count in init_transform_pcfg if a rule has more than one transformation.
    """
    t_count = 0
    for program in program.programs:
        for rule in program.rules:
            if len(rule.transforms) > 1:
                t_count += 1
    return t_count

def t_extract_rules_from_ast(node, rules_count, token_rules_count, current_transform=None):
    if isinstance(node, list):
        for item in node:
            print("item:", item)
            t_extract_rules_from_ast(item, rules_count, token_rules_count, current_transform)
    elif isinstance(node, dict):
        for child in node.children:
            t_extract_rules_from_ast(child, rules_count, token_rules_count, current_transform)
    elif hasattr(node, '__dataclass_fields__'):
        class_name = node.__class__.__name__
        print("class-name:", class_name)
        if class_name in transform_operations:
            current_transform = class_name
            if any(class_name == rule for rule in transform_rules['Transform']):
                rules_count[class_name] += 1
        for field_name, _ in node.__dataclass_fields__.items():
            field_value = getattr(node, field_name)
            if isinstance(field_value, str) and current_transform:
                token_rules_count[(current_transform, field_value)] += 1
            else:
                t_extract_rules_from_ast(field_value, rules_count, token_rules_count, current_transform)
        if class_name in transform_operations:
            current_transform = None
    elif isinstance(node, str) and current_transform:
        token_rules_count[(current_transform, node)] += 1

# Step 2: Probability estimation based on counts
def compute_probabilities(rules_count, token_rules_count, token_type_counts):
    probabilities = {}
    total = sum(rules_count.values())
    for rule, count in rules_count.items():
        if rule == 'Not' or rule == 'Or' or rule == 'And' or rule == 'VarAnd':
            division = f"{count} / 4"
            probabilities[rule] = count / total
        else:
            division = f"{count} / {total}"
            probabilities[rule] = count / total # TODO: more fine-grained
        print(f"Rule: {rule}, Count: {count}, Division: {division}, Probability: {probabilities[rule]}")

    for ((token_type, token_value)), count in token_rules_count.items():
        total_tokens_of_type = token_type_counts[token_type]
        division = f"{count} / {total_tokens_of_type}"
        probabilities[(token_type, token_value)] = count / total_tokens_of_type
        print(f"Token: {(token_type, token_value)}, Count: {count}, Division: {division}, Total Tokens of Type: {total_tokens_of_type}, Probability: {probabilities[(token_type, token_value)]}")
    return probabilities

def laplace_smoothing_transforms(computed_probabilities, alpha=1):
    smoothed_probabilities = defaultdict(dict)
    # Handle transform rules
    print("computed_probabilities:", computed_probabilities)
    total_transforms = sum(value for key, value in computed_probabilities.items() if isinstance(key, str))
    total_transform_rules = len(init_transform_pcfg['Transform'])

    for rule, _ in init_transform_pcfg['Transform'].items():
        computed_count = computed_probabilities.get(str(rule), 0)
        smoothed_count = computed_count + alpha
        total_smoothed_count = total_transforms + alpha * total_transform_rules
        smoothed_probabilities['Transform'][rule] = round(smoothed_count / total_smoothed_count, 2)
        print(f"Transform Rule: {rule}, Computed Count: {computed_count}, Smoothed Count: {smoothed_count}, Total Smoothed Count: {total_smoothed_count}, Smoothed Probability: {smoothed_probabilities['Transform'][rule]}")

    for category, rules in init_transform_pcfg.items():
        if category == 'Transform':
            continue
        total_tokens_of_type = sum(value for key, value in computed_probabilities.items() if isinstance(key, tuple) and key[0] == category)
        total_category_rules = len(rules)
        for rule, _ in rules.items():
            computed_count = computed_probabilities.get((category, rule), 0)
            smoothed_count = computed_count + alpha
            total_smoothed_count = total_tokens_of_type + alpha * total_category_rules
            smoothed_probabilities[category][rule] = round(smoothed_count / total_smoothed_count, 2)
            print(f"Transform Category: {category}, Rule: {rule}, Computed Count: {computed_count}, Smoothed Count: {smoothed_count}, Total Smoothed Count: {total_smoothed_count}, Smoothed Probability: {smoothed_probabilities[category][rule]}")
    return smoothed_probabilities

def compute_transform_costs(taskNumber):
    ast_program = processtask(taskNumber)
    print("ast:", ast_program)
    transform_rules_count, t_token_rules_count, t_token_type_counts = defaultdict(int), defaultdict(int), defaultdict(int)
    print("init_transform_pcfg:", init_transform_pcfg)
    t_extract_rules_from_ast(ast_program, transform_rules_count, t_token_rules_count)
    transform_rules_count['Transforms'] = increment_transforms(ast_program, init_transform_pcfg)
    print("transform_rules_count:", transform_rules_count)

    print("token_rules_count:", t_token_rules_count) # Count of non-terminals
    for ((token_type, _)), count in t_token_rules_count.items():
        t_token_type_counts[token_type] += count
    print("t_token_type_counts:", t_token_type_counts)
    t_probabilities = compute_probabilities(transform_rules_count, t_token_rules_count, t_token_type_counts)
    print("pre-smoothing-t_probabilities:", t_probabilities)
    t_smoothed_probabilities = laplace_smoothing_transforms(t_probabilities, alpha=1)
    print("smoothed_probs:", t_smoothed_probabilities)
    return t_smoothed_probabilities

taskNumber = "6855a6e4"
compute_transform_costs(taskNumber)
# todo: variable
##--------------------- Computing filter probabilities ---------------------------------------------------------
filter_operations = {'FilterByColor', 'FilterBySize', 'FilterByDegree', 'FilterByNeighborColor', 'FilterByNeighborSize', 'FilterByNeighborDegree', 
                    'FilterByShape', 'FilterByColumns', 'FilterByHeight', 'FilterByRows', 'FilterByWidth', 'Or', 'And', 'Not', 'VarAnd'}

# Filter grammar rules
filter_rules = {
    'Filters': [
        'Not',
        'And',
        'Or',
        'VarAnd'
    ],
    'Filter': [
        'FilterByColor',
        'FilterBySize',
        'FilterByDegree',
        'FilterByHeight',
        'FilterByShape',
        'FilterByColumns',
        #FilterByRows,
        #FilterByWidth
        'FilterByNeighborColor',
        'FilterByNeighborSize',
        'FilterByNeighborDegree',
    ],
    'FColor': [
        'O', 'B', 'R', 'G', 'Y', 'X', 'F', 'A', 'C', 'W', 'most', 'least'
    ],
    'Shape': ['enclosed', 'square'],
    'Size': size_values,
    'Degree': degree_values,
    'Relation': ['IsAnyNeighbor', 'IsDirectNeighbor', 'IsDiagonalNeighbor'],
    'Height': height_values,
    'Width': width_values,
    'Column': column_values,
    'Row': row_values
}
init_filter_pcfg = initialize_uniform_pcfg(filter_rules)

def f_extract_rules_from_ast(node, rules_count, token_rules_count, current_filter=None):
    if isinstance(node, list):
        for item in node:
            f_extract_rules_from_ast(item, rules_count, token_rules_count, current_filter)
    elif isinstance(node, dict):
        for child in node.children:
            f_extract_rules_from_ast(child, rules_count, token_rules_count, current_filter)
    elif hasattr(node, '__dataclass_fields__'):
        class_name = node.__class__.__name__
        if class_name in filter_operations:
            current_filter = class_name
            if any(class_name == rule for rule in filter_rules['Filter']):
                rules_count[class_name] += 1
        for field_name, _ in node.__dataclass_fields__.items():
            field_value = getattr(node, field_name)
            if isinstance(field_value, str) and current_filter:
                token_rules_count[(current_filter, field_value)] += 1
            else:
                f_extract_rules_from_ast(field_value, rules_count, token_rules_count, current_filter)
        if class_name in filter_operations:
            current_filter = None
    elif isinstance(node, str) and current_filter:
        token_rules_count[(current_filter, node)] += 1

def laplace_smoothing_for_filters(computed_probabilities, alpha=1):
    smoothed_probabilities = defaultdict(dict)
    # Handle filters and their sub-rules
    for category, rules in init_filter_pcfg.items():
        if category in ['FColor', 'Size', 'Degree', 'Shape', 'Row', 'Column', 'Height', 'Width', 'Relation']:  # Skip token categories
            continue
        total_category_counts = sum(computed_probabilities.get((category, rule_key), 0) for rule_key in rules.keys())
        total_category_rules = len(rules)
        for rule, _ in rules.items():
            computed_count = computed_probabilities.get((str(rule)), 0)
            smoothed_count = computed_count + alpha
            total_smoothed_count = total_category_counts + alpha * total_category_rules
            smoothed_probabilities[category][rule] = round(smoothed_count / total_smoothed_count, 2)
            print(f"P('{rule}' in '{category}') = {smoothed_count}/{total_smoothed_count} = {smoothed_probabilities[category][rule]:.2f}")
            print(f"Filter: {rule}, Category: {category}, Computed Count: {computed_count}, Smoothed Count: {smoothed_count}, Total Smoothed Count: {total_smoothed_count}, Smoothed Probability: {smoothed_probabilities[category][rule]}")

    # Handle token categories (Color, Size, Degree, Shape)
    for token_category, token_values in init_filter_pcfg.items():
        print("token-category", token_category)
        if token_category not in ['FColor', 'Size', 'Degree', 'Shape', 'Row', 'Column', 'Height', 'Width', 'Relation']:
            continue
        # Correctly iterating over token values for the current category
        total_tokens_of_type = sum(computed_probabilities.get((token_category, token_value), 0) for token_value in token_values)
        total_token_rules = len(token_values)
        
        for token_value in token_values:  # Iterate over values in the category, not keys of the entire PCFG
            computed_count = computed_probabilities.get((token_category, token_value), 0)
            smoothed_count = computed_count + alpha
            total_smoothed_count = total_tokens_of_type + alpha * total_token_rules
            smoothed_probabilities[token_category][token_value] = round(smoothed_count / total_smoothed_count, 2)
            print(f"FToken: {token_value}, Category: {token_category}, Computed Count: {computed_count}, Smoothed Count: {smoothed_count}, Total Smoothed Count: {total_smoothed_count}, Smoothed Probability: {smoothed_probabilities[token_category][token_value]:.2f}")
    return smoothed_probabilities
    
def compute_filter_costs(taskNumber):
    ast_program = processtask(taskNumber)
    print("ast:", ast_program)
    filter_rules_count, f_token_rules_count, f_token_type_counts = defaultdict(int), defaultdict(int), defaultdict(int)
    print("init_filter_pcfg:", init_filter_pcfg)
    f_extract_rules_from_ast(ast_program, filter_rules_count, f_token_rules_count)
    print("filter_rules_count:", filter_rules_count)
    print("token_rules_count:", f_token_rules_count) # Count of non-terminals
    for ((token_type, _)), count in f_token_rules_count.items():
        f_token_type_counts[token_type] += count
    print("f_token_type_counts:", f_token_type_counts)
    f_probabilities = compute_probabilities(filter_rules_count, f_token_rules_count, f_token_type_counts)
    print("pre-smoothing-f_probabilities:", f_probabilities)
    f_smoothed_probabilities = laplace_smoothing_for_filters(f_probabilities, alpha=1)
    print("smoothed_probs:", f_smoothed_probabilities)
    return f_smoothed_probabilities

compute_filter_costs(taskNumber)