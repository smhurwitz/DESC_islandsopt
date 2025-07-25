import os
import re
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../"))
import csv

from desc.compute import all_kwargs, data_index


def _escape(line):
    match = re.findall(r"\|.*\|", line)
    if match:
        sub = r"\|" + match[0][1:-1] + "|"
        line = line.replace(match[0], sub)
    return line


def write_csv(parameterization):
    with open(parameterization + ".csv", "w", newline="") as f:
        fieldnames = [
            "Name",
            "Label",
            "Units",
            "Description",
            "Aliases",
            "kwargs",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        datidx = data_index[parameterization]
        kwargs = all_kwargs[parameterization]
        keys = datidx.keys()
        for key in sorted(keys):
            if key not in data_index[parameterization][key]["aliases"]:
                if not data_index[parameterization][key]["public"]:
                    continue
                d = {
                    "Name": "``" + key + "``",
                    "Label": ":math:`" + datidx[key]["label"].replace("$", "") + "`",
                    "Units": datidx[key]["units_long"],
                    "Description": datidx[key]["description"],
                    "Aliases": f"{['``' + alias + '``' for alias in datidx[key]['aliases']]}".strip(
                        "[]"
                    ).replace(
                        "'", ""
                    ),
                    "kwargs": ", ".join(
                        ["``" + str(k) + "``" for k in kwargs[key].keys()]
                    ),
                }
                # stuff like |x| is interpreted as a substitution by rst, need to escape
                d["Description"] = _escape(d["Description"])
                writer.writerow(d)


header = r"""
List of Variables
#################

The table below contains a list of variables that are used in the code and that are
available for plotting / analysis.

  * **Name** : Name of the variable as it appears in the code. Pass a string with this
    name to any of the plotting functions to plot, or to the relevant ``.compute()``
    method to return the calculated quantity.
  * **Label** : TeX label for the variable.
  * **Units** : Physical units for the variable.
  * **Description** : Description of the variable.
  * **Aliases** : Alternative names of the variable that are equivalent to the primary
    name.
  * **kwargs** : Optional keyword arguments that can be supplied when computing the
    variable. See the bottom of this page for detailed descriptions and default values
    of each argument. The only keyword argument that is valid for all variables is
    'basis' (see explanation below).

All vector quantities are computed in toroidal coordinates :math:`(R,\phi,Z)` by default.
The keyword argument ``basis='xyz'`` can be used to convert the variables into Cartesian
coordinates :math:`(X,Y,Z)`. ``basis`` must be one of ``{'rpz', 'xyz'}``.

Our convention to denote partial derivatives is an underscore followed by the first
letter of the coordinate that the partial derivative is taken with respect to. Unless
otherwise specified or implied by the variable name, these partial derivatives are
those of the DESC :math:`\rho, \theta, \zeta` coordinate system. For example, ``|B|_z``
is :math:`(\partial \vert B \vert / \partial\zeta)|_{\rho, \theta}`.

Many quantities require special grids to compute accurately.
To not burden users with such bookkeeping,
when an object method such as ``eq.compute(...,override_grid=True)`` is called,
DESC will automatically use a set of best grids for the computation.
However, when writing objectives developers must perform the bookkeeping
and ensure everything can be computed accurately on the chosen grid.

"""

block = """

{}
{}

.. csv-table:: List of Variables: {}
   :file: {}.csv
   :widths: 23, 15, 15, 60, 15, 15
   :header-rows: 1

"""

for parameterization in data_index.keys():
    if len(data_index[parameterization]):
        write_csv(parameterization)
        header += block.format(
            parameterization,
            "-" * len(parameterization),
            parameterization,
            parameterization,
        )

kwargs_description = """
Optional Keyword arguments
--------------------------

.. list-table:: kwargs
   :widths: 25 100
   :header-rows: 1

   * - Name
     - Description
"""
unique_kwargs = {}
for p in all_kwargs.keys():
    for name in all_kwargs[p]:
        if all_kwargs[p][name]:
            unique_kwargs.update(all_kwargs[p][name])
for k in sorted(unique_kwargs):
    kwargs_description += f"""
   * - ``{k}``
     - {unique_kwargs[k]}
"""

header += kwargs_description

with open("variables.rst", "w+") as f:
    f.write(header)
