#!/bin/bash

wget -P ./data "https://zenodo.org/records/8386059/files/arcade_challenge_datasets.zip"
unzip ./data/arcade_challenge_datasets.zip -d ./data
mv ./data/arcade_challenge_datasets ./data/ARCADE
