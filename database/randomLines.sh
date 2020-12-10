#!/bin/bash

#randomly create 3 kinds of images: 
#overlapping lines (in input/), top line (in topLine/), bottom line (in bottomLine/)

#Parameters
start=0
totalFiles=1000
dirOut="./validSetLines/"
dir1="${dirOut}inputGray/"
dir2="${dirOut}topLine/"
dir3="${dirOut}bottomLine/"
dir4="${dirOut}target/"
builddir="/tmp/"

#Create the directories if they don't exist
if ! [ -e "${dirOut}" ]
then
  mkdir "${dirOut}"
fi

if ! [ -e "${builddir}" ]
then
  mkdir "${builddir}"
fi

if ! [ -e "${dir1}" ]
then
  mkdir "${dir1}"
fi

if ! [ -e "${dir2}" ]
then
  mkdir "${dir2}"
fi

if ! [ -e "${dir3}" ]
then
  mkdir "${dir3}"
fi

if ! [ -e "${dir4}" ]
then
  mkdir "${dir4}"
fi

#Loop until all desired documents are created
nfile=${start}
while [ "$nfile" -le "$totalFiles" ] 
do
  filename=$(printf %05d ${nfile}) #zero-padded number for images name
  
  #Create documents headlines
  for FILE in "${builddir}input.tex" "${builddir}top.tex" "${builddir}bot.tex"
  do
echo "\documentclass[10pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[french]{babel}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\begin{document}
\pagenumbering{gobble}" > "${FILE}"
  done

  #Randomly create 2 lines
  lines[0]=''
  lines[1]=''
  for n in 0 1
  do
    NUMBER=$(cat /dev/urandom | tr -dc '0-9' | fold -w 256 | head -n 1 | sed -e 's/^0*//' | head --bytes 2)
    if [ "$NUMBER" == "" ]; then
      NUMBER=0
    fi
    text=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9 ' | fold -w $(($NUMBER/2)) | head -n 1)

    #Clever loop ! See the link below 
    #https://stackoverflow.com/questions/10551981/how-to-perform-a-for-loop-on-each-character-in-a-string-in-bash
    #Add spaces to the random text
    lastNumber=10
    i=0
    while IFS='' read -n 1 c
    do
      lines[n]="${lines[n]}${c}"
      if [ "$c" = ' ' ]
      then
        i=0
      fi
      if [[ *"$c"* == "123456789" ]]
      then
        lastNumber="$c"
      fi
      if [ $i -ge $lastNumber ]
      then
        lines[n]="${lines[n]} "
        i=0
      fi
      i=$((${i}+1))
    done < <(printf %s "$text")
  done
  
  #Randomly selects the position of the lines in the page and the distance between the two lines
  #height=$(cat /dev/urandom | tr -dc '0-9' | fold -w 256 | head -n 1 | sed -e 's/^0*//' | head --bytes 2)
  #if [ "$NUMBER" == "" ]; then
   #height=0
  #fi
  spacing=$(cat /dev/urandom | tr -dc '3-7' | fold -w 256 | head -n 1 | sed -e 's/^0*//' | head --bytes 1)

#Creates the correct lines in the 3 documents
#\vspace*{$((${height}*2/5))\baselineskip}
echo "
\center{${lines[0]}}\\\\

\vspace{-1.${spacing}\baselineskip}
\center{${lines[1]}}
\end{document}" >> "${builddir}input.tex"

echo "
\center{${lines[0]}}\\\\
\end{document}" >> "${builddir}top.tex"

echo "
\vspace*{-1.${spacing}\baselineskip}
\center{${lines[1]}}
\end{document}" >> "${builddir}bot.tex"

  #Alternative method to get an image. Crop around the lines.
  #latex -halt-on-error -output-directory="${dir}${builddir}" "${dir}${filename}.tex" > /dev/null
  #dvipng "${dir}${builddir}${filename}.dvi" -o "${dir}${filename}.png" > /dev/null

  #Generate input image
  latex -halt-on-error -interaction=batchmode -output-format='pdf' -output-directory="${builddir}" "${builddir}input.tex" > /dev/null
  gs -sDEVICE=png16m -sOutputFile="${dir1}${filename}.png" -dBATCH -dNOPAUSE -r300 -dTextAlphaBits=4 \
-dGraphicsAlphaBits=4 -dUseTrimBox -dDownScaleFactor=3 -f "${builddir}input.pdf" > /dev/null
  convert "${dir1}${filename}.png" -colorspace gray -crop 256x256+280+50 "${dir1}${filename}.png"

  #Generate top line image
  latex -halt-on-error -interaction=batchmode -output-format='pdf' -output-directory="${builddir}" "${builddir}top.tex" > /dev/null
  gs -sDEVICE=png16m -sOutputFile="${dir2}${filename}.png" -dBATCH -dNOPAUSE -r300 -dTextAlphaBits=4 \
-dGraphicsAlphaBits=4 -dUseTrimBox -dDownScaleFactor=3 -f "${builddir}top.pdf" > /dev/null 
  convert "${dir2}${filename}.png" -colorspace gray -crop 256x256+280+50 "${dir2}${filename}.png"
  
  #Generate bottom line image
  latex -halt-on-error -interaction=batchmode -output-format='pdf' -output-directory="${builddir}" "${builddir}bot.tex" > /dev/null
  gs -sDEVICE=png16m -sOutputFile="${dir3}${filename}.png" -dBATCH -dNOPAUSE -r300 -dTextAlphaBits=4 \
-dGraphicsAlphaBits=4 -dUseTrimBox -dDownScaleFactor=3 -f "${builddir}bot.pdf" > /dev/null
  convert "${dir3}${filename}.png" -colorspace gray -crop 256x256+280+50 "${dir3}${filename}.png"

  convert "${dir2}${filename}.png" "${dir3}${filename}.png" -negate "${dir1}${filename}.png" -channel RGB -combine "${dir4}${filename}.png"

  nfile=$((${nfile}+1))
done
