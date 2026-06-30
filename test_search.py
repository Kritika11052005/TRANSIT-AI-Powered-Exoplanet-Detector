import lightkurve as lk

tic_id = 100100727
print("1. Search with author='SPOC' and exptime=120:")
try:
    sr1 = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC", exptime=120)
    print(f"   Found: {len(sr1)} entries")
except Exception as e:
    print(f"   Error: {e}")

print("2. Search with author='SPOC' (no exptime restriction):")
try:
    sr2 = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC")
    print(f"   Found: {len(sr2)} entries")
    if len(sr2) > 0:
        print(f"   Authors: {set(sr2.author)}, Exptimes: {set(sr2.exptime)}")
except Exception as e:
    print(f"   Error: {e}")

print("3. Search for any TESS light curve:")
try:
    sr3 = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
    print(f"   Found: {len(sr3)} entries")
    if len(sr3) > 0:
        print(f"   Authors: {set(sr3.author)}, Exptimes: {set(sr3.exptime)}")
except Exception as e:
    print(f"   Error: {e}")
