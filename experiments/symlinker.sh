#THIS IS A SIMPLE COMMAND LINE UTILITY SCRIPT THAT HELPS MAKE SYMBOLIC LINKS FOR RUNGROUPS THAT ARE SPREAD ACROSS TIME AND SPACE
rg_name=$0
date1=$1
date2=$2
for p in $(ls -d /proj/smallfry/embeddings/$1-$0 && ls -d /proj/smallfry/embeddings/$2-$0); do
    joinDir=$($(dirname $p)/merged-$0)
    mkdir -p ${joinDir}
    for oldRunDir in $(ls -d $p/*); do
        unlink $joinDir/$(filename $oldRunDir)
        ln -s -f $oldRunDir $joinDir
    done
done