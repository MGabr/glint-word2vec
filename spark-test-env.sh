#!/bin/sh

# taken and adapted from https://github.com/unchartedsoftware/salt-core
# see also https://uncharted.software/blog/continuous-integration-with-apache-spark/

WORKDIR=`pwd | rev | cut -d "/" -f1 | rev`

RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
BLUE='\033[0;94m'
RESET='\033[0;00m'

create_test_environment() {
  printf "${GREEN}Creating${RESET} new Spark test environment (container: ${BLUE}${WORKDIR}${RESET})...\n"
  docker run \
  --name $WORKDIR \
  -p 8080:8080 \
  -p 9999:9999 \
  -p 50070:50070 \
  -d \
  -v /`pwd`/src/it/resources/log4j.properties:/opt/spark-2.4.4-bin-hadoop2.6/conf/log4j.properties \
  -v /`pwd`:`pwd` \
  -it \
  --shm-size=1g \
  --workdir="/`pwd`" \
  mgabr/sparklet-hdfs:2.4.4 bash
  sleep 30
}

run_test_environment() {
  printf "${GREEN}Resuming${RESET} existing Spark test environment (container: ${BLUE}${WORKDIR}${RESET})...\n"
  docker start $WORKDIR
}

stop_test_environment() {
  printf "${YELLOW}Stopping${RESET} Spark test environment (container: ${BLUE}${WORKDIR}${RESET})...\n"
  docker stop $WORKDIR
}

kill_test_environment() {
  printf "${RED}Destroying${RESET} Spark test environment (container: ${BLUE}${WORKDIR}${RESET})...\n"
  docker rm -fv $WORKDIR
}

attach_test_environment() {
  printf "${GREEN}Attaching${RESET} to Spark test environment (container: ${BLUE}${WORKDIR}${RESET})...\n"
  docker attach $WORKDIR
}

exec_test_environment() {
    printf "${GREEN}Executing${RESET} in Spark test environment (container: ${BLUE}${WORKDIR}${RESET})...\n"
    docker exec $WORKDIR $@
}

exec_detach_test_environment() {
    printf "${GREEN}Executing detached${RESET} in Spark test environment (container: ${BLUE}${WORKDIR}${RESET})...\n"
    docker exec --detach $WORKDIR $@
}

verify_test_environment() {
  PRESENT=$(docker ps -a -q -f name=$WORKDIR)
  if [ -n "$PRESENT" ]; then
    run_test_environment
  else
    create_test_environment
  fi
}

if [ "$1" = "stop" ]; then
  stop_test_environment
elif [ "$1" = "rm" ]; then
  kill_test_environment
elif [ "$1" = "attach" ]; then
  attach_test_environment
elif [ "$1" = "exec" ]; then
  shift
  exec_test_environment $@
elif [ "$1" = "exec-detach" ]; then
  shift
  exec_detach_test_environment $@
else
  verify_test_environment
fi