#!/usr/bin/awk -f
# Keep only the header and rows where the "visited" column equals 1.
# Usage: awk -f filter_visited.awk pool.csv > filtered.csv
#    or: awk -f filter_visited.awk pool.csv -i inplace  (gawk only)
BEGIN { FS = OFS = "," }
NR == 1 {
    for (i = 1; i <= NF; i++)
        if ($i == "visited") { col = i; break }
    print; next
}
$col == 1
