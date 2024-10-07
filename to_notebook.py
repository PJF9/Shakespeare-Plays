from src.utils.to_notebook import py_to_ipynb

import os


def transformer_experiment() -> None:
    # Output notebook path
    output_ipynb = 'transformer_experiment.ipynb'

    # Add title to the notebook
    py_to_ipynb(
        py_files=[''],
        comment = '# Transformer Experiment',
        output_ipynb=output_ipynb
    )

    # Add classes differentiation markdown
    py_to_ipynb(
        py_files=[''],
        comment = '## src',
        output_ipynb=output_ipynb
    )

    # Add configs to the notebook
    py_to_ipynb(
        py_files=['src/config.py'],
        comment = '### Configs',
        output_ipynb=output_ipynb
    )

    # Add utils to the notebook
    py_to_ipynb(
        py_files=[f'src/utils/{file}' for file in os.listdir('src/utils')],
        comment = '### Utils',
        output_ipynb=output_ipynb
    )

    # Add datasets to the notebook
    py_to_ipynb(
        py_files=['src/dataset/transformer_dataset.py'],
        comment='### Dataset Class',
        output_ipynb=output_ipynb
    )

    # Add evaluator to the notebook
    py_to_ipynb(
        py_files=['src/evaluator/transformer_evaluator.py'],
        comment='### Evaluator Class',
        output_ipynb=output_ipynb
    )

    # Add trainer to the notebook
    py_to_ipynb(
        py_files=['src/training/transformer_trainer.py'],
        comment='### Trainer Class',
        output_ipynb=output_ipynb
    )
    
    # Add generators to the notebook
    py_to_ipynb(
        py_files=['src/models/generators.py'],
        comment='### Models',
        output_ipynb=output_ipynb
    )

    # Add training script
    py_to_ipynb(
        py_files=['train_transformer.py'],
        comment='## Train',
        output_ipynb=output_ipynb
    )

    # Add evaluating script
    py_to_ipynb(
        py_files=['evaluate_transformer.py'],
        comment='## Evaluate',
        output_ipynb=output_ipynb
    )

    # Add evaluating script
    py_to_ipynb(
        py_files=['generate_transformer.py'],
        comment='## Generate',
        output_ipynb=output_ipynb
    )


def lstm_experiment() -> None:
    # Output notebook path
    output_ipynb = 'lstm_experiment.ipynb'

    # Add title to the notebook
    py_to_ipynb(
        py_files=[''],
        comment = '# LSTM Experiment',
        output_ipynb=output_ipynb
    )

    # Add classes differentiation markdown
    py_to_ipynb(
        py_files=[''],
        comment = '## src',
        output_ipynb=output_ipynb
    )

    # Add configs to the notebook
    py_to_ipynb(
        py_files=['src/config.py'],
        comment = '### Configs',
        output_ipynb=output_ipynb
    )

    # Add utils to the notebook
    py_to_ipynb(
        py_files=[f'src/utils/{file}' for file in os.listdir('src/utils')],
        comment = '### Utils',
        output_ipynb=output_ipynb
    )

    # Add datasets to the notebook
    py_to_ipynb(
        py_files=['src/dataset/lstm_dataset.py'],
        comment='### Dataset Class',
        output_ipynb=output_ipynb
    )

    # Add evaluator to the notebook
    py_to_ipynb(
        py_files=['src/evaluator/lstm_evaluator.py'],
        comment='### Evaluator Class',
        output_ipynb=output_ipynb
    )

    # Add trainer to the notebook
    py_to_ipynb(
        py_files=['src/training/lstm_trainer.py'],
        comment='### Trainer Class',
        output_ipynb=output_ipynb
    )
    
    # Add generators to the notebook
    py_to_ipynb(
        py_files=['src/models/generators.py'],
        comment='### Models',
        output_ipynb=output_ipynb
    )

    # Add training script
    py_to_ipynb(
        py_files=['train_lstm.py'],
        comment='## Train',
        output_ipynb=output_ipynb
    )

    # Add evaluating script
    py_to_ipynb(
        py_files=['evaluate_lstm.py'],
        comment='## Evaluate',
        output_ipynb=output_ipynb
    )

    # Add evaluating script
    py_to_ipynb(
        py_files=['generate_lstm.py'],
        comment='## Generate',
        output_ipynb=output_ipynb
    )


if __name__ == '__main__':
    experiment_name = 'LSTM'

    if experiment_name == 'LSTM':
        lstm_experiment()
    else:
        transformer_experiment()
