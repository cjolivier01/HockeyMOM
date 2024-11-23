function get_tag {
  if [ -f "./tag" ]; then
    TAG="$(cat ./tag)"
  fi
  echo $TAG
}
