function get_tag {
  if [ -f "./tag" ]; then
    TAG="$(cat ./tag)"
  fi
  echo $TAG
}


function is_arm {
  ARM="$(cat /proc/cpuinfo  | grep "model name" | grep ARM | awk '{print$4}')"
  if [ -z "${ARM}" ]; then
    echo "0"
  fi
  echo "1"
}
