# from src.doc_intel_client import analyze_pdf_from_url
# filepath = "https://test0files0rud.blob.core.windows.net/files/Long-Lease-Final.pdf"
# sastoken = "sp=r&st=2025-11-25T17:22:27Z&se=2025-11-26T01:37:27Z&spr=https&sv=2024-11-04&sr=b&sig=EIYSSstXWDNgqbaT%2Fd7%2BcL3Xr6fTRYFnoGP5cAgrjS8%3D"
# sas = f"{filepath}?{sastoken}"

# res = analyze_pdf_from_url(sas, model="prebuilt-contract")
# print(res.paragraphs[0].content)


from src.pipelines.ingest_pipeline import ingest_contract

r = ingest_contract("/Users/rudra/Downloads/Long-Lease-Final.pdf")
print(r["doc_id"])
print(len(r["chunks"]))
