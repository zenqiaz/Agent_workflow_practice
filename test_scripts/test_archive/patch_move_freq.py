"""Move run_freq_job out of the commented-out block in server_with_product.py."""

with open("server_with_product.py", "r") as f:
    content = f.read()

# The run_freq_job code is currently inside a '''...''' block.
# Strategy: remove it from there and insert it before the ''' marker.

# 1. Find and extract the run_freq_job function text
freq_start = "@mcp.tool()\nasync def run_freq_job("
freq_end_marker = '    return json.dumps({\n        "status": "ok",\n        "label": job_label,\n        "energy": energy,\n        "enthalpy_eh": enthalpy,\n        "gibbs_free_energy_eh": gibbs,\n        "product": "gibbs_free_energy_eh",\n    })\n\n\n'

idx_start = content.index(freq_start)
idx_end = content.index(freq_end_marker, idx_start) + len(freq_end_marker)
freq_code = content[idx_start:idx_end]

# 2. Remove it from current position
content = content[:idx_start] + content[idx_end:]

# 3. Find the ''' that starts the comment block (it's right after run_sp_energy return)
# The pattern is: ...product": "energy"})\n'''\n
sp_end = '    return json.dumps({"status": "ok", "label": job_label, "energy": energy, "product": "energy"})\n'
insert_pos = content.index(sp_end) + len(sp_end)

# The ''' should be right at insert_pos
assert content[insert_pos:insert_pos+3] == "'''", f"Expected ''' at position, got: {repr(content[insert_pos:insert_pos+20])}"

# 4. Insert freq_code before the '''
content = content[:insert_pos] + "\n\n" + freq_code + content[insert_pos:]

with open("server_with_product.py", "w") as f:
    f.write(content)
print("OK: moved run_freq_job before the commented-out block")
