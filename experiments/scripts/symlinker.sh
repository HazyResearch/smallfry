#THIS IS A SIMPLE COMMAND LINE UTILITY SCRIPT THAT HELPS MAKE SYMBOLIC LINKS FOR RUNGROUPS THAT ARE SPREAD ACROSS TIME AND SPACE
rg_name=$1
for p in $(ls -d /lfs/1/tginart/proj/smallfry/embeddings/*-$1); do
    joinDir=$(dirname $p)/merged-$1
    mkdir -p ${joinDir}
    for oldRunDir in $(ls -d $p/*); do
        ln -s -f $oldRunDir $joinDir
    done
done
