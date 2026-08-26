#!/usr/bin/env bash
set -e
python -m src.prepare_data
python -m src.train --model image
python -m src.train --model text
python -m src.train --model fusion
python -m src.evaluate --model image
python -m src.evaluate --model text
python -m src.evaluate --model fusion
