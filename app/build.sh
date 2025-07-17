#!/bin/bash

sudo apt-get install python3-dev default-libmysqlclient-dev build-essential


# Install Python dependencies
pip3.9 install --disable-pip-version-check --target . --upgrade -r /vercel/path0/requirements.txt

# Run Django collectstatic command
python3 manage.py collectstatic --no-input
