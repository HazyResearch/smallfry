#THIS IS A SIMPLE COMMAND LINE UTILITY SCRIPT THAT HELPS MAKE SYMBOLIC LINKS FOR RUNGROUPS THAT ARE SPREAD ACROSS TIME AND SPACE
rg_name=$1
date1=$2
date2=$3
for p in $(ls -d /proj/smallfry/embeddings/$2-$1 && ls -d /proj/smallfry/embeddings/$3-$1); do
    joinDir=$(dirname $p)/merged-$1
    mkdir -p ${joinDir}
    for oldRunDir in $(ls -d $p/*); do
        ln -s -f $oldRunDir $joinDir
    done
done
