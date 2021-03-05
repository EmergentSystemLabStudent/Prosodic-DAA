#!/bin/bash

python3 prosodyextruct.py

python3 compress.py
rm -rf graph/
python3 compress_delta.py
rm -rf graph/
python3 compress_delta_delta.py
rm -rf graph/

python3 compatibility_transform.py
rm -rf results/*.npz
