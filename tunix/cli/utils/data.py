import ast
from asyncio import streams
import importlib
import functools
import grain
import os


def apply_chat_template(x, tokenizer):
  print(x)
  return {
      "prompts": tokenizer.apply_chat_template(
          x["prompt"], tokenize=False, add_generation_prompt=True
      ),
      **{k: v for k, v in x.items() if k != "prompt"},
  }


def get_dataset_from_parquet(parquet_path, tokenizer):
  dataset = grain.experimental.ParquetIterDataset(parquet_path)
  return dataset.map(functools.partial(apply_chat_template, tokenizer=tokenizer))


def parse_call_string(arg_string):
  if not arg_string.strip():
    return [], {}

  fake_expression = f"dummy_func({arg_string})"
  try:
    tree = ast.parse(fake_expression)
  except SyntaxError:
    raise ValueError(f"Invalid argument syntax: {arg_string}")

  call_node = tree.body[0].value
  
  parsed_args = []
  for node in call_node.args:
    parsed_args.append(ast.literal_eval(node))

  parsed_kwargs = {}
  for keyword in call_node.keywords:
    parsed_kwargs[keyword.arg] = ast.literal_eval(keyword.value)

  return parsed_args, parsed_kwargs


def get_dataset_from_module(specifier: str, tokenizer):
  """Get dataset from module.

  Examples of specifier:
    - "data.coding"
    - "data.coding:get_dataset"
    - "data.coding:get_my_dataset"
    - "data.coding:get_dataset(name='coding_v0')"
    - "data.coding:get_dataset('coding_v0', split='train')"
    - "/home/user/project/data/coding.py:get_dataset"

  Args:
    specifier: The specifier of the module.
    tokenizer: The tokenizer to apply to the dataset.
  Returns:
    The dataset.
  """
  if '(' in specifier and ':' in specifier:
    specifier, args_part = specifier.rsplit("(", 1)
  else:
    args_part = ''
  if ':' in specifier:
    specifier, func_spec = specifier.rsplit(":", 1)
  else:
    func_spec = ''
  if os.path.exists(specifier) and specifier.endswith(".py"):
    module_name = os.path.splitext(os.path.basename(specifier))[0]
    spec = importlib.util.spec_from_file_location(module_name, specifier)
    module = importlib.util.module_from_spec(spec)
    try:
      spec.loader.exec_module(module)
    except Exception as e:
      raise ImportError(
          f"Failed to execute module {module_name} from {specifier}: {e}"
      )
  else:
    module = importlib.import_module(specifier)
  args = []
  kwargs = {}
  if func_spec:
    func = getattr(module, func_spec)
    if args_part:
      args_part = args_part.rstrip(')')
      args, kwargs = parse_call_string(args_part)
      
  else:
    func = module.get_dataset
  dataset = func(*args, **kwargs)
  return dataset.map(functools.partial(apply_chat_template, tokenizer=tokenizer))
