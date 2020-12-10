#!/bin/bash

#Simple script to zeropad file names
for i in {0..5000}
do
  x=$(printf %05d ${i})
  mv "${i}.png" "$x.png"
done
