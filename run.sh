CheckProcess()
{

  if [ "$1" = "" ];
  then
    return 1
  fi
 

  PROCESS_NUM=`ps -ef | grep "$1" | grep -v "grep" | wc -l` 
  if [ $PROCESS_NUM -eq 2 ];
  then
    return 0
  else
    return 1
  fi
}
 
 

while [ 1 ] ; do
 CheckProcess "janus"
 CheckQQ_RET=$?
 if [ $CheckQQ_RET -eq 1 ];
 then 
  sudo killall -9 janus
  exec sudo janus &  
 fi
 sleep 1
done


