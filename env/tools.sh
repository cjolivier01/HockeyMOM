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

increment_tag_minor_version() {
    # Define the file path
    local file_path="minor_version"

    # Read the number from the file
    read -r number < "$file_path"

    # Increment the number
    number=$((10#$number + 1))

    # Format the number as a zero-prefixed four-digit number and write it back to the file
    printf -v new_number "%04d" $number
    echo $new_number > "$file_path"
    echo $new_number
}
