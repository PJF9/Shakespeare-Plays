import json

import os
from typing import List, Union, Dict, Any


def __load_notebook(notebook_path: str) -> Dict[str, Any]:
    '''
    Loads an existing Jupyter notebook.
    
    Args:
        notebook_path (str): Path to the .ipynb file to load.
        
    Returns:
        dict: The notebook content as a dictionary.
    '''
    if os.path.exists(notebook_path):
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
    else:
        # If the file doesn't exist, create an empty notebook structure
        notebook = {
            'cells': [],
            'metadata': {
                'kernelspec': {
                    'display_name': 'Python 3',
                    'language': 'python',
                    'name': 'python3'
                },
                'language_info': {
                    'codemirror_mode': {
                        'name': 'ipython',
                        'version': 3
                    },
                    'file_extension': '.py',
                    'mimetype': 'text/x-python',
                    'name': 'python',
                    'nbconvert_exporter': 'python',
                    'pygments_lexer': 'ipython3',
                    'version': '3.x'
                }
            },
            'nbformat': 4,
            'nbformat_minor': 2
        }
    return notebook


def py_to_ipynb(
        py_files: List[str],
        output_ipynb: str,
        comment: Union[str, None]=None
    ) -> None:
    '''
    Converts a list of Python (.py) files into a Jupyter Notebook (.ipynb) with optional comments.
    
    Args:
        py_files (list of str): List of paths to the .py files.
        output_ipynb (str): Path to the output .ipynb file.
        comment (list of str): The comment markdown of the notebook.
    '''
    # Load the existing notebook or create a new one
    notebook = __load_notebook(output_ipynb)

    # Add a Markdown cell for the comment if provided
    if comment:
        markdown_cell = {
            'cell_type': 'markdown',
            'metadata': {},
            'source': comment
        }
        notebook['cells'].append(markdown_cell)  # Append the markdown cell to the notebook

    # Iterate through each .py file and corresponding comment
    for py_file in py_files:
        if py_file.endswith('__.py') or not py_file.endswith('.py'):
            continue

        # Read the Python (.py) file content
        with open(py_file, 'r') as f:
            source_code = f.read()
        
        # Create a code cell for the Python script content
        code_cell = {
            'cell_type': 'code',
            'metadata': {},
            'source': source_code.splitlines(True),  # Split lines to maintain formatting
            'outputs': [],
            'execution_count': None
        }
        
        # Append the code cell to the list of cells
        notebook['cells'].append(code_cell)
    
    
    # Write the notebook to a file
    with open(output_ipynb, 'w') as f:
        json.dump(notebook, f, indent=4)
