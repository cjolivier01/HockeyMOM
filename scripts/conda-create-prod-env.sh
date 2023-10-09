#!/bin/bash
PROD_ENV="ubuntu-prod"
mamba env remove --name "${PROD_ENV}"
mamba create --name "${PROD_ENV}" --clone ubuntu
