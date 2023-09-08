#!/bin/bash

search_dir=.
for entry in "$search_dir"/*
do
  qsub $entry
done
