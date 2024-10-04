import os
import socket


def _print_slurm_environment():
    if "SLURM_PROCID" in os.environ and int(os.environ["SLURM_PROCID"]) == 0:
        for key, val in os.environ.items():
            if key.startswith("SLURM_"):
                print(f"{key}={val}")


# _print_slurm_environment()
# exit(0)

node_lists = [
    "clip-g1-[01-03]",
    "clip-g1-[0-1],clip-g2-[2-3]",
    "clip-g1-0,clip-g2-0",
    "clip-g1-0,clip-g2-1",
    "clip-g1-1",
    "clip-a-[1,3,5]",
    "clip-b-[1-3,5]",
    "clip-c-[1-3,5,9-12]",
    "clip-d-[5,9-12]",
    "clip-e-[5,9],clip-e-[15-19]",
    "clip-f-[5,9],clip-f-[15,17]",
    "clip-f-5,clip-f-[15,17]",
    "clip-f-[5,9],clip-f-175",
]


def set_slurm_env_variables(node_list, tasks_per_node):
    """
    Set virtual Slurm environment variables based on the node list and tasks per node.

    Args:
    - node_list (list of str): List of node names
    - tasks_per_node (int): Number of tasks per node
    """
    # Setting SLURM_JOB_NODELIST and SLURM_TASKS_PER_NODE
    os.environ["SLURM_JOB_NODELIST"] = ",".join(node_list)
    os.environ["SLURM_TASKS_PER_NODE"] = str(tasks_per_node)

    # Setting SLURM_JOB_NUM_NODES
    os.environ["SLURM_JOB_NUM_NODES"] = str(len(node_list))

    # Setting SLURM_NNODES to be the same as SLURM_JOB_NUM_NODES (for some scripts)
    os.environ["SLURM_NNODES"] = os.environ["SLURM_JOB_NUM_NODES"]

    # Setting SLURM_NTASKS as total number of tasks
    os.environ["SLURM_NTASKS"] = str(len(node_list) * tasks_per_node)

    # Mock other possible Slurm environment variables as needed...


def add_string_numbers(a: str, b: str) -> str:
    # Step 2: Make the strings of equal length by prefixing zeros
    max_len = max(len(a), len(b))
    a = a.rjust(max_len, "0")
    b = b.rjust(max_len, "0")

    result = []
    carry = 0

    # Step 3: Start adding from the rightmost digit
    for i in range(max_len - 1, -1, -1):
        total = carry + int(a[i]) + int(b[i])
        carry = total // 10
        result.append(str(total % 10))

    # Step 4: If there's any carry left, add it to the front of the result
    if carry:
        result.append(str(carry))

    # Combine the result and reverse to get the proper order
    return "".join(result[::-1])


def slurm_parse_int(s):
    for i, c in enumerate(s):
        if c not in "0123456789":
            return s[:i], s[i:]
    return int(s), ""


def string_range(a, b):
    results = [a]
    next_num = a
    while int(next_num) + 1 != int(b):
        next_num = add_string_numbers(next_num, "1")
        results.append(next_num)
    return results


def slurm_parse_brackets(s):
    # parse a "bracket" expression (including closing ']')
    lst = []
    while len(s) > 0:
        if s[0] == ",":
            s = s[1:]
            continue
        if s[0] == "]":
            return lst, s[1:]
        a, s = slurm_parse_int(s)
        assert len(s) > 0, f"Missing closing ']'"
        if s[0] in ",]":
            lst.append(a)
        elif s[0] == "-":
            b, s = slurm_parse_int(s[1:])
            lst += string_range(a, int(b) + 1)
    assert len(s) > 0, f"Missing closing ']'"


def slurm_parse_node(s):
    # parse a "node" expression
    for i, c in enumerate(s):
        if c == ",":  # name,...
            return [s[:i]], s[i + 1 :]
        if c == "[":  # name[v],...
            b, rest = slurm_parse_brackets(s[i + 1 :])
            if len(rest) > 0:
                assert rest[0] == ",", f"Expected comma after brackets in {s[i:]}"
                rest = rest[1:]
            return [s[:i] + str(z) for z in b], rest

    return [s], ""


def slurm_parse_list(s):
    lst = []
    while len(s) > 0:
        v, s = slurm_parse_node(s)
        lst.extend(v)
    return lst


# for s in node_lists:
#     print(s)
#     print(slurm_parse_list(s))


def _get_first_hostname(nodelist):
    nodelist = slurm_parse_list(os.environ.get("SLURM_STEP_NODELIST", ""))
    if not nodelist:
        return None
    print(f"Master: {nodelist[0]}")
    return nodelist[0]


def get_dist_url(hostname, port=29500, protocol="tcp"):
    if get_num_machines() < 2:
        return None
    """Generate a PyTorch dist-url using the given hostname and port."""
    ip = socket.gethostbyname(hostname)
    os.environ["MASTER_PORT"] = f"{port}"
    os.environ["MASTER_ADDR"] = ip
    return f"{protocol}://{ip}:{port}"


def get_default_dist_url():
    if "MASTER_ADDR" in os.environ:
        assert "MASTER_PORT" in os.environ
        return get_dist_url(
            os.environ["MASTER_ADDR"], port=int(os.environ["MASTER_PORT"])
        )

    nodelist = os.environ.get("SLURM_JOB_NODELIST", None)
    if not nodelist:
        return None
    master = _get_first_hostname(nodelist)
    if master:
        return get_dist_url(master)
    return None


def get_local_rank():
    lr = int(os.environ.get("SLURM_LOCALID", "0"))
    # os.environ["LOCAL_RANK"] = str(lr)
    return lr


def get_machine_rank():
    return int(os.environ.get("SLURM_NODEID", "0"))


def get_num_machines():
    
    num_machines = int(os.environ.get("SLURM_NNODES", "1"))
    # os.environ["WORLD_SIZE"] = str(num_machines)
    return num_machines
