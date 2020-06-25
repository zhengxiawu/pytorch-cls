for x in `ls *.tar`
do
  filename=`basename $x .tar`
  mkdir $filename
  tar -xvf $x -C ./$filename && rm -rf $filename.tar
done
