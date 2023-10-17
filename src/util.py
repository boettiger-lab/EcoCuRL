""" GENERAL UTILITIES WITHOUT OTHER NATURAL PLACE TO BE """

from collections.abc import Iterable
from numbers import Number

def iter_pretty_print(iterable_obj):
  if (len(iterable_obj) < 11) or (len(iterable_obj) > 500):
    print("[", *[ 
      f"{x:.3f}, " 
      if (isinstance(x, Number))
      else f"{x}, "
      for x in iterable_obj 
      ] ,"]")
  else:
    print("[")
    idx=0
    for x in iterable_obj:
      if (isinstance(x, Number)):
        print(f"{x:.3f}, ", end="")
      else:
        print(f"{x}, ", end="")
      if idx % 10 ==0:
        print("\n")
      idx+=1
    print("]")

def dict_pretty_print(D: dict, indent_lvl: int = 0, indent_size: int = 2, verbose:bool =False, dict_name: str=""):
  """ RStudio terminal doesn't like json.dumps so do it manual """
  preamble = dict_name
  if len(preamble)>0:
    preamble += " = "
  if (indent_lvl == 0):
    if verbose:
      print("Using 3 decimal places.")
    print(preamble)
  base_indent = indent_lvl * " "
  indent = (indent_lvl + indent_size) * " "
  print(base_indent + "{")
  for key, value in D.items():
    print(f"{indent}{key}: ", end="")
    if isinstance(value, dict):
      print("")
      dict_pretty_print(D=value, indent_lvl = indent_lvl + indent_size, indent_size=indent_size)
    elif isinstance(value, Iterable) and (not isinstance(value, str)):
      iter_pretty_print(value)
    elif isinstance(value, Number):
      print(f"{value:.3f}")
    else:
      print(value)
  print(base_indent + "}")

def print_params(env) -> None:
  """ Pretty-prints the parameters  of the env. (env is a gym class with a env.config['parameters'] attribute.) """
  dict_pretty_print(env.env_dyn_obj.dyn_params.param_vals())