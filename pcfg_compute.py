from arga_ast_generator import *
from arga_ast_generator import _Ast, _Filter_Op
from collections import defaultdict
from lark import Tree, Token
from task import *
from VocabMaker import *
from filters import *
import enum

taskNumber = "ea32f347"
task = Task("ARC/data/training/" + taskNumber + ".json")
task.abstraction = "nbccg"
task.input_abstracted_graphs_original[task.abstraction] = [getattr(input, Image.abstraction_ops[task.abstraction])() for
                                                        input in task.train_input]
task.output_abstracted_graphs_original[task.abstraction] = [getattr(output, Image.abstraction_ops[task.abstraction])() for
                                                        output in task.train_output]
task.get_static_inserted_objects()
task.get_static_object_attributes(task.abstraction)
setup_size_and_degree_based_on_task(task)
vocabMakers = [FColor, Size, Degree, FilterByColor, FilterBySize, FilterByDegree, FilterByNeighborColor, FilterByNeighborSize, FilterByNeighborDegree, Or, And]

vocab = VocabFactory.create(vocabMakers)
size_values, degree_values = [], []

for leaf in list(vocab.leaves()):
    if isinstance(leaf, SizeValue) and not isinstance(leaf, enum.Enum):
        size_values.append(str(leaf.value))
    elif isinstance(leaf, DegreeValue) and not isinstance(leaf, enum.Enum):
        degree_values.append(str(leaf.value))

with open("dsl/dsl.lark", "r") as f:
    arga_dsl_grammar = f.read()
ast_parser = Lark(arga_dsl_grammar, start="start", parser="lalr", transformer=ToAst())
with open(f"dsl/gens/gens_20231120/{taskNumber}_correct.txt", "r") as f:
    program = f.read()
ast_program = ast_parser.parse(program)

# Step 1: PCFG initialization and the extraction of counts from program ASTs
def initialize_uniform_pcfg(grammar_rules):
    pcfg = defaultdict(dict)
    for non_terminal, productions in grammar_rules.items():
        uniform_prob = 1.0 / len(productions)
        for production in productions:
            pcfg[non_terminal][production] = uniform_prob
    return pcfg

# Transform grammar rules
transform_rules = {
    'Transform': [
        ('UpdateColor', 'COLOR'),
        ('MoveNode', 'DIRECTION'),
        ('ExtendNode', 'DIRECTION', 'OVERLAP'),
        ('MoveNodeMax', 'DIRECTION'),
        ('RotateNode', 'ROT_ANGLE'),
        ('AddBorder', 'COLOR'),
        ('FillRectangle', 'COLOR', 'OVERLAP'),
        ('HollowRectangle', 'COLOR'),
        ('Mirror', 'MIRROR_AXIS'),
        ('Flip', 'SYMMETRY_AXIS')
    ],
    'COLOR': [
        'O', 'B', 'R', 'G', 'Y', 'X', 'F', 'A', 'C', 'W'
    ],
    'DIRECTION': ['U', 'D', 'L', 'R', 'UL', 'UR', 'DL', 'DR'],
    'OVERLAP': ['True', 'False'],
    'ROTATION_ANGLE': ['90', '180', '270'],
    'SYMMETRY_AXIS': ['VERTICAL', 'HORIZONTAL', 'DIAGONAL_LEFT', 'DIAGONAL_RIGHT'],
    'MIRROR_AXIS': [('X', None), (None, 'Y'), ('X', 'Y')]
}

# Filter grammar rules
filter_rules = {
    'Filters': [
        ('Filter', ''),
        ('Not', 'Filter'),
        ('And', 'Filter', 'Filter'),
        ('Or', 'Filter', 'Filter')
    ],
    'Filter': [
        ('FilterByColor', 'COLOR'),
        ('FilterBySize', 'SIZE'),
        ('FilterByDegree', 'DEGREE'),
        ('FilterByNeighborColor', 'COLOR'),
        ('FilterByNeighborSize', 'SIZE'),
        ('FilterByNeighborDegree', 'DEGREE')
    ],
    'COLOR': [
        'O', 'B', 'R', 'G', 'Y', 'X', 'F', 'A', 'C', 'W'
    ],
    'SIZE': size_values,
    'DEGREE': degree_values,
}

init_transform_pcfg = initialize_uniform_pcfg(transform_rules)
init_filter_pcfg = initialize_uniform_pcfg(filter_rules)
transform_rules_count, filter_rules_count = defaultdict(int), defaultdict(int)
t_token_rules_count, f_token_rules_count = defaultdict(int), defaultdict(int)

def t_extract_rules_from_ast(node, rules_count, token_rules_count, current_transform=None):
    if isinstance(node, list):
        for item in node:
            t_extract_rules_from_ast(item, rules_count, token_rules_count, current_transform)
    elif isinstance(node, Tree):
        for child in node.children:
            t_extract_rules_from_ast(child, rules_count, token_rules_count, current_transform)
    elif hasattr(node, '__dataclass_fields__'):
        class_name = node.__class__.__name__
        if class_name in transform_operations:
            current_transform = class_name
            rules_count[class_name] += 1
        for field_name, field_info in node.__dataclass_fields__.items():
            field_value = getattr(node, field_name)
            if isinstance(field_value, Token) and current_transform:
                token_type = field_value.type
                token_rules_count[(token_type, field_value.value)] += 1
            else:
                t_extract_rules_from_ast(field_value, rules_count, token_rules_count, current_transform)
        if class_name in transform_operations:
            current_transform = None
    elif isinstance(node, Token) and current_transform:
        token_type = node.type
        token_rules_count[(token_type, node.value)] += 1

def f_extract_rules_from_ast(node, rules_count, token_rules_count, current_filter=None):
    if isinstance(node, list):
        for item in node:
            f_extract_rules_from_ast(item, rules_count, token_rules_count, current_filter)
    elif isinstance(node, Tree):
        for child in node.children:
            f_extract_rules_from_ast(child, rules_count, token_rules_count, current_filter)
    elif hasattr(node, '__dataclass_fields__'):
        class_name = node.__class__.__name__
        if class_name in filter_operations:
            current_filter = class_name
            rules_count[class_name] += 1
        for field_name, field_info in node.__dataclass_fields__.items():
            field_value = getattr(node, field_name)
            if isinstance(field_value, Token) and current_filter:
                token_type = field_value.type
                token_rules_count[(token_type, field_value.value)] += 1
            else:
                f_extract_rules_from_ast(field_value, rules_count, token_rules_count, current_filter)
        if class_name in filter_operations:
            current_filter = None
    elif isinstance(node, Token) and current_filter:
        token_type = node.type
        token_rules_count[(token_type, node.value)] += 1

transform_operations = {'UpdateColor', 'HollowRectangle', 'FillRectangle', 'AddBorder', 'MoveNode',
                        'ExtendNode', 'MoveNodeMax', 'Mirror', 'Flip', 'RotateNode', 'NoOp'}
filter_operations = {'FilterByColor', 'FilterBySize', 'FilterByDegree', 'FilterByNeighborColor',
                     'FilterByNeighborSize', 'FilterByNeighborDegree', 'Or', 'And', 'Not'}

t_extract_rules_from_ast(ast_program, transform_rules_count, t_token_rules_count)
f_extract_rules_from_ast(ast_program, filter_rules_count, f_token_rules_count)

t_token_type_counts, f_token_type_counts = defaultdict(int), defaultdict(int)
for ((token_type, _)), count in t_token_rules_count.items():
    t_token_type_counts[token_type] += count

for ((token_type, _)), count in f_token_rules_count.items():
    f_token_type_counts[token_type] += count

# Step 2: Probability estimation based on counts
def compute_probabilities(rules_count, token_rules_count, token_type_counts):
    probabilities = {}
    total = sum(rules_count.values())
    for rule, count in rules_count.items():
        if rule == 'Not' or rule == 'Or' or rule == 'And':
            division = f"{count} / 4"
            probabilities[rule] = count / total
        else:
            division = f"{count} / {total}"
            probabilities[rule] = count / total # TODO: more fine-grained
        #print(f"Rule: {rule}, Count: {count}, Division: {division}, Probability: {probabilities[rule]}")

    for ((token_type, token_value)), count in token_rules_count.items():
        total_tokens_of_type = token_type_counts[token_type]
        division = f"{count} / {total_tokens_of_type}"
        probabilities[(token_type, token_value)] = count / total_tokens_of_type
        #print(f"Token: {(token_type, token_value)}, Count: {count}, Division: {division}, Total Tokens of Type: {total_tokens_of_type}, Probability: {probabilities[(token_type, token_value)]}")

    return probabilities

# Step 3: LaPlace Smoothing
def laplace_smoothing(alpha=1):
    computed_probabilities = compute_probabilities(transform_rules_count, t_token_rules_count, t_token_type_counts)
    smoothed_probabilities = defaultdict(dict)
    # Handle transform rules
    
    total_transforms = sum(value for key, value in computed_probabilities.items() if isinstance(key, str))
    total_transform_rules = len(init_transform_pcfg['Transform'])

    for rule, initial_prob in init_transform_pcfg['Transform'].items():
        computed_count = computed_probabilities.get(str(rule[0]), 0)
        smoothed_count = computed_count + alpha
        total_smoothed_count = total_transforms + alpha * total_transform_rules
        smoothed_probabilities['Transform'][rule] = round(smoothed_count / total_smoothed_count, 2)
        #print(f"Transform Rule: {rule}, Computed Count: {computed_count}, Smoothed Count: {smoothed_count}, Total Smoothed Count: {total_smoothed_count}, Smoothed Probability: {smoothed_probabilities['Transform'][rule]}")

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
            #print(f"Transform Category: {category}, Rule: {rule}, Computed Count: {computed_count}, Smoothed Count: {smoothed_count}, Total Smoothed Count: {total_smoothed_count}, Smoothed Probability: {smoothed_probabilities[category][rule]}")
    return smoothed_probabilities

from collections import defaultdict
import pprint

pp = pprint.PrettyPrinter(indent=4)

def laplace_smoothing_for_filters(alpha=1):
    smoothed_probabilities = defaultdict(dict)
    computed_probabilities = compute_probabilities(filter_rules_count, f_token_rules_count, f_token_type_counts)
    # Handle filters and their sub-rules
    for category, rules in init_filter_pcfg.items():
        if category in ['COLOR', 'SIZE', 'DEGREE']:  # Skip token categories
            continue
        total_category_counts = sum(computed_probabilities.get((category, rule_key), 0) for rule_key in rules.keys())
        total_category_rules = len(rules)
        for rule, initial_prob in rules.items():
            computed_count = computed_probabilities.get((str(rule[0])), 0)
            smoothed_count = computed_count + alpha
            total_smoothed_count = total_category_counts + alpha * total_category_rules
            smoothed_probabilities[category][rule] = round(smoothed_count / total_smoothed_count, 2)
            #print(f"P('{rule}' in '{category}') = {smoothed_count}/{total_smoothed_count} = {smoothed_probabilities[category][rule]:.2f}")
            #print(f"Filter: {rule}, Category: {category}, Computed Count: {computed_count}, Smoothed Count: {smoothed_count}, Total Smoothed Count: {total_smoothed_count}, Smoothed Probability: {smoothed_probabilities[category][rule]}")

    # Handle token categories (COLOR, SIZE, DEGREE)
    for token_category, token_values in init_filter_pcfg.items():
        if token_category not in ['COLOR', 'SIZE', 'DEGREE']:
            continue
        total_tokens_of_type = sum(computed_probabilities.get((token_category, token_value), 0) for token_value in token_values.keys())
        total_token_rules = len(token_values)
        for token_value in token_values.keys():
            computed_count = computed_probabilities.get((token_category, token_value), 0)
            smoothed_count = computed_count + alpha
            total_smoothed_count = total_tokens_of_type + alpha * total_token_rules
            smoothed_probabilities[token_category][token_value] = round(smoothed_count / total_smoothed_count, 2)
            #print(f"P('{token_value}' in '{token_category}') = {smoothed_count}/{total_smoothed_count} = {smoothed_probabilities[token_category][token_value]:.2f}")
            #print(f"Token: {token_value}, Category: {token_category}, Computed Count: {computed_count}, Smoothed Count: {smoothed_count}, Total Smoothed Count: {total_smoothed_count}, Smoothed Probability: {smoothed_probabilities[token_category][token_value]}")
    return smoothed_probabilities

t_probabilities = compute_probabilities(transform_rules_count, t_token_rules_count, t_token_type_counts)

t_smoothed_probabilities = laplace_smoothing()
f_smoothed_probs = laplace_smoothing_for_filters()