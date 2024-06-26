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
gens_dir = "dsl/v0_3/generations/gpt4o_20240514/"

# get the AST program from the generations directory
def test_file(filename, parser, xformer):
    with open(filename, "r") as f:
        lib = "(" + f.read() + ")"
    ##print(f"Testing {filename}...")
    t = parser.lib_parse_tree(lib)
    ast = xformer.transform(t)
    pprint(ast)
    return ast

Color._sizes = {color.name: 1 for color in Color}
Dir._sizes = {dir.name: 1 for dir in Dir}
Overlap._sizes = {overlap.name: 1 for overlap in Overlap}
Rotation_Angle._sizes = {overlap.name: 1 for overlap in Rotation_Angle}
Symmetry_Axis._sizes = {overlap.name: 1 for overlap in Symmetry_Axis}
RelativePosition._sizes = {relativepos.name: 1 for relativepos in RelativePosition}
ImagePoints._sizes = {imagepts.name: 1 for imagepts in ImagePoints}
Mirror_Axis._sizes = {mirror_axis.name: 1 for mirror_axis in Mirror_Axis}

FColor._sizes = {color.name: 1 for color in FColor}
Size._sizes = {s.value: 1 for s in Size._all_values}
Degree._sizes = {degree.value: 1 for degree in Degree._all_values}
Shape._sizes = {shape.name: 1 for shape in Shape}
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
    vocabMakers = [ObjectId, FColor, Degree, Height, Width, Size, Shape, Row, Column, 
                Neighbor_Of, Color_Equals, Size_Equals, Degree_Equals, Shape_Equals, Height_Equals,
                Column_Equals, Not, 
                And, Or]
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
    file_path = f"dsl/v0_3/generations/gpt4o_20240514/{taskNumber}_valid_programs.txt"
    ast_program = test_file(file_path, parser, xformer)
    #print(ast_program)
    return ast_program

##--------------------- Computing transform probabilities ---------------------------------------------------------
transform_operations = {'UpdateColor', 'HollowRectangle', 'FillRectangle', 'AddBorder', 'MoveNode', 
                        'ExtendNode', 'MoveNodeMax', 'Mirror', 'Flip', 'RotateNode', 'NoOp', 'Insert', 
                        'Transforms'}
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
        'O', 'B', 'R', 'G', 'Y', 'X', 'F', 'A', 'C', 'W', 'most', 'least', 'VarColor'
    ],
    'Direction': ['UP', 'DOWN', 'LEFT', 'RIGHT', 'UP_LEFT', 'UP_RIGHT', 'DOWN_LEFT', 'DOWN_RIGHT', 'VarDirection'],
    'Overlap': ['True', 'False'],
    'Rotation_Angle': ['90', '180', '270'],
    'Symmetry_Axis': ['VERTICAL', 'HORIZONTAL', 'DIAGONAL_LEFT', 'DIAGONAL_RIGHT'],
    'ImagePoints': ['TOP', 'BOTTOM', 'LEFT', 'RIGHT', 'TOP_LEFT', 'TOP_RIGHT', 'BOTTOM_LEFT', 'BOTTOM_RIGHT', 'VarImagePoints'],
    'RelativePosition': ['SOURCE', 'TARGET', 'MIDDLE'],
    'Mirror_Axis': ['VarMirror'],
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

def increment_transforms(program):
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
            t_extract_rules_from_ast(item, rules_count, token_rules_count, current_transform)
    elif isinstance(node, dict):
        for child in node.children:
            t_extract_rules_from_ast(child, rules_count, token_rules_count, current_transform)
    elif hasattr(node, '__dataclass_fields__'):
        class_name = node.__class__.__name__
        current_transform = class_name  # Setting current transform to this class
        if class_name in transform_operations:
            current_transform = class_name
            if any(class_name == rule for rule in transform_rules['Transform']):
                rules_count[class_name] += 1
        for field_name, value in node.__dataclass_fields__.items():
            field_value = getattr(node, field_name)
            if str(field_value).startswith('DirectionOf'):
                token_rules_count[('Direction', ("VarDirection"))] += 1
            elif field_name == "color" and str(field_value).startswith('ColorOf'):
                token_rules_count[('Color', str("VarColor"))] += 1
            elif str(field_value).startswith('ImagePointsOf'):
                token_rules_count[('ImagePoints', str("VarImagePoints").lower())] += 1
            elif str(field_value).startswith('MirrorAxisOf'):
                token_rules_count[('Mirror_Axis', str("VarMirror").lower())] += 1
            elif not str(field_value).startswith('Var') and \
                (isinstance(field_value, str) or isinstance(field_value, bool) or isinstance(field_value, int)) and current_transform:
                token_rules_count[(class_name, str(field_value))] += 1
                ##print(f"Counting Token: {(class_name, field_value)}")  # Debug #print
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
        if rule == 'Not' or rule == 'Or' or rule == 'And':
            division = f"{count} / 3"
            probabilities[rule] = count / total
        else:
            division = f"{count} / {total}"
            probabilities[rule] = count / total
        ##print(f"Rule: {rule}, Count: {count}, Division: {division}, Probability: {probabilities[rule]}")

    for ((token_type, token_value)), count in token_rules_count.items():
        total_tokens_of_type = token_type_counts[token_type]
        division = f"{count} / {total_tokens_of_type}"
        probabilities[(token_type, token_value)] = count / total_tokens_of_type
        ##print(f"Token: {(token_type, token_value)}, Count: {count}, Division: {division}, Total Tokens of Type: {total_tokens_of_type}, Probability: {probabilities[(token_type, token_value)]}")
    return probabilities

def laplace_smoothing_transforms(computed_probabilities, alpha=1):
    smoothed_probabilities = defaultdict(dict)
    # Handle transform rules
    init_transform_pcfg = initialize_uniform_pcfg(transform_rules)
    total_transforms = sum(value for key, value in computed_probabilities.items() if isinstance(key, str))
    total_transform_rules = len(init_transform_pcfg['Transform'])
    for rule, _ in init_transform_pcfg['Transform'].items():
        computed_count = computed_probabilities.get(rule, 0)
        smoothed_count = computed_count + alpha
        total_smoothed_count = total_transforms + alpha * total_transform_rules
        smoothed_probabilities['Transform'][rule] = round(smoothed_count / total_smoothed_count, 2)
        ##print(f"Transform Rule: {rule}, Computed Count: {computed_count}, Smoothed Count: {smoothed_count}, Total Smoothed Count: {total_smoothed_count}, Smoothed Probability: {smoothed_probabilities['Transform'][rule]}")

    for category, rules in init_transform_pcfg.items():
        if category == 'Transform':
            continue
        total_tokens_of_type = sum(value for key, value in computed_probabilities.items() if isinstance(key, tuple) and key[0] == category)
        total_category_rules = len(rules)
        for rule, _ in rules.items():
            computed_count = computed_probabilities.get((category, str(rule)), 0)
            smoothed_count = computed_count + alpha
            total_smoothed_count = total_tokens_of_type + alpha * total_category_rules
            smoothed_probabilities[category][str(rule)] = round(smoothed_count / total_smoothed_count, 2)
            ##print(f"Transform Category: {category}, Rule: {str(rule)}, Computed Count: {computed_count}, Smoothed Count: {smoothed_count}, Total Smoothed Count: {total_smoothed_count}, Smoothed Probability: {smoothed_probabilities[category][rule]}")
    return smoothed_probabilities

def compute_transform_costs(taskNumber):
    ast_program = processtask(taskNumber)
    transform_rules_count, t_token_rules_count, t_token_type_counts = defaultdict(int), defaultdict(int), defaultdict(int)
    t_extract_rules_from_ast(ast_program, transform_rules_count, t_token_rules_count)
    transform_rules_count['Transforms'] = increment_transforms(ast_program)
    print("token_rules_count:", t_token_rules_count) # Count of non-terminals
    print("transform_rules_count:", transform_rules_count)
    for ((token_type, _)), count in t_token_rules_count.items():
        t_token_type_counts[token_type] += count
    print("t_token_type_counts:", t_token_type_counts)
    t_probabilities = compute_probabilities(transform_rules_count, t_token_rules_count, t_token_type_counts)
    ##print("pre-smoothing-t_probabilities:", t_probabilities)
    t_smoothed_probabilities = laplace_smoothing_transforms(t_probabilities, alpha=1)
    ##print("smoothed_probs:", t_smoothed_probabilities)
    return t_smoothed_probabilities

#compute_transform_costs(taskNumber)

##--------------------- Computing filter probabilities ---------------------------------------------------------
filter_operations = {'Color_Equals', 'Size_Equals', 'Degree_Equals', 'Shape_Equals', 'Neighbor_Color', 'Neighbor_Size', 'Neighbor_Degree', 'Neighbor_Of', 'Direct_Neighbor_Of',
                    'Shape_Equals', 'Column_Equals', 'Height_Equals', 'Row_Equals', 'Width_Equals', 'Or', 'And', 'Not'}

# Filter grammar rules
filter_rules = {
    'Filters': [
        'Not',
        'And',
        'Or'
    ],
    'Filter': [
        'Color_Equals',
        'Size_Equals',
        'Degree_Equals',
        'Height_Equals',
        'Shape_Equals',
        "Column_Equals",
        'Row_Equals',
        'Width_Equals',
        'Neighbor_Of',
        'Direct_Neighbor_Of'
    ],
    'FColor': [
        'O', 'B', 'R', 'G', 'Y', 'X', 'F', 'A', 'C', 'W', 'most', 'least', 'ColorOf'
    ],
    'Shape': ['enclosed', 'square', 'ShapeOf'],
    'Size': size_values,
    'Degree': degree_values,
    'Height': height_values,
    'Width': width_values,
    'Column': column_values,
    'Row': row_values
}

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
            if str(field_value).startswith('ColorOf'):
                token_rules_count[('FColor', str("ColorOf"))] += 1
            elif str(field_value).startswith('SizeOf'):
                token_rules_count[('Size', str('SizeOf'))] += 1
            elif str(field_value).startswith('HeightOf'):
                token_rules_count[('Height', str('HeightOf'))] += 1
            elif str(field_value).startswith('WidthOf'):
                token_rules_count[('Width', str('WidthOf'))] += 1
            elif str(field_value).startswith('RowOf'):
                token_rules_count[('Row', str('RowOf'))] += 1
            elif str(field_value).startswith('ColumnOf'):
                token_rules_count[('Column', str('ColumnOf'))] += 1
            elif str(field_value).startswith('ShapeOf'):
                token_rules_count[('Shape', str('ShapeOf'))] += 1
            elif str(field_value).startswith('DegreeOf'):
                token_rules_count[('Degree', str('DegreeOf'))] += 1
            elif isinstance(field_value, str) and current_filter:
                token_rules_count[(class_name, field_value)] += 1 # todo: this
            else:
                f_extract_rules_from_ast(field_value, rules_count, token_rules_count, current_filter)
        if class_name in filter_operations:
            current_filter = None
    elif isinstance(node, str) and current_filter:
        token_rules_count[(current_filter, node)] += 1

def laplace_smoothing_for_filters(computed_probabilities, alpha=2):
    smoothed_probabilities = defaultdict(dict)
    # Handle filters and their sub-rules
    init_filter_pcfg = initialize_uniform_pcfg(filter_rules)
    ##print("computed_probabilities:", computed_probabilities)
    for category, rules in init_filter_pcfg.items():
        if category in ['FColor', 'Size', 'Degree', 'Shape', 'Row', 'Column', 'Height', 'Width']:  # Skip token categories
            continue
        total_category_counts = sum(computed_probabilities.get((category, rule_key), 0) for rule_key in rules.keys())
        total_category_rules = len(rules)
        for rule, _ in rules.items():
            computed_count = computed_probabilities.get((str(rule)), 0)
            smoothed_count = computed_count + alpha
            total_smoothed_count = total_category_counts + alpha * total_category_rules
            smoothed_probabilities[category][rule] = round(smoothed_count / total_smoothed_count, 2)
            ##print(f"P('{rule}' in '{category}') = {smoothed_count}/{total_smoothed_count} = {smoothed_probabilities[category][rule]:.2f}")
            ##print(f"Filter: {rule}, Category: {category}, Computed Count: {computed_count}, Smoothed Count: {smoothed_count}, Total Smoothed Count: {total_smoothed_count}, Smoothed Probability: {smoothed_probabilities[category][rule]}")

    for token_category, token_values in init_filter_pcfg.items(): # todo: FColor
        if token_category not in ['FColor', 'Size', 'Degree', 'Shape', 'Row', 'Column', 'Height', 'Width']:
            continue
        # Correctly iterating over token values for the current category
        total_tokens_of_type = sum(computed_probabilities.get((token_category, token_value), 0) for token_value in token_values)
        total_token_rules = len(token_values)
        
        for token_value in token_values:  # Iterate over values in the category, not keys of the entire PCFG
            computed_count = computed_probabilities.get((token_category, token_value), 0)
            smoothed_count = computed_count + alpha
            total_smoothed_count = total_tokens_of_type + alpha * total_token_rules
            smoothed_probabilities[token_category][token_value] = round(smoothed_count / total_smoothed_count, 2)
            ##print(f"FToken: {token_value}, Category: {token_category}, Computed Count: {computed_count}, Smoothed Count: {smoothed_count}, Total Smoothed Count: {total_smoothed_count}, Smoothed Probability: {smoothed_probabilities[token_category][token_value]:.2f}")
    return smoothed_probabilities
    
def compute_filter_costs(taskNumber):
    ast_program = processtask(taskNumber)
    ##print("ast:", ast_program)
    filter_rules_count, f_token_rules_count, f_token_type_counts = defaultdict(int), defaultdict(int), defaultdict(int)
    f_extract_rules_from_ast(ast_program, filter_rules_count, f_token_rules_count)
    print("filter_rules_count:", filter_rules_count)
    print("f_token_type_counts:", f_token_type_counts)

    print("token_rules_count:", f_token_rules_count) # Count of non-terminals
    for ((token_type, _)), count in f_token_rules_count.items():
        f_token_type_counts[token_type] += count
    f_probabilities = compute_probabilities(filter_rules_count, f_token_rules_count, f_token_type_counts)
    f_smoothed_probabilities = laplace_smoothing_for_filters(f_probabilities, alpha=1)
    ##print("smoothed_probs:", f_smoothed_probabilities)
    return f_smoothed_probabilities

#print(compute_transform_costs("ae3edfdc"))
##print(compute_filter_costs("ae3edfdc"))