import sys


def write_dict_in_columns(data_dict, out_file, table_width: int):
    column_width = (table_width - 2) // 2
    # Create list of formatted key-value strings
    kv_pairs = [f"{key}: {value}"[:column_width] for key, value in data_dict.items()]

    # Ensure the list has an even number of elements
    if len(kv_pairs) % 2 != 0:
        kv_pairs.append("")

    # Calculate the number of rows needed for two columns
    num_rows = len(kv_pairs) // 2

    # Prepare horizontal border and empty row format
    border_top = "┌" + "─" * column_width + "┬" + "─" * column_width + "┐\n"
    row_format = "│{:<" + str(column_width) + "}│{:<" + str(column_width) + "}│\n"
    border_bottom = "└" + "─" * column_width + "┴" + "─" * column_width + "┘\n"

    # Print the top border
    out_file.write(border_top)

    # Print each row
    for i in range(num_rows):
        left = kv_pairs[i]
        right = kv_pairs[i + num_rows]
        out_file.write(row_format.format(left, right))

    # Print the bottom border
    out_file.write(border_bottom)


# Example dictionary
data = {
    "key1": "This is a long value that might need truncation because it exceeds the maximum length",
    "key2": "Value2",
    "key3": "Value3",
    "key4": "Value4",
    "key5": "Another long value that will get cut off somewhere around here",
}

write_dict_in_columns(data, out_file=sys.stdout, table_width=80)
sys.stdout.flush()
