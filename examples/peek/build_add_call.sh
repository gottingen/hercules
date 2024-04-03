#!/bin/bash

# This script builds the add_call example.

gcc -o call_add call_add.c -L. -ladd_shared -Wl,-rpath=.
g++ -o call_add_cc call_add.cc -L. -ladd_shared -Wl,-rpath=.