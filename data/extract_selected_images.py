import os
import zipfile
import pandas as pd

# -----------------------------
# 1. Paths
# -----------------------------

csv_path = r"data\selected_images.csv"

zip1 = r"C:\Users\hgspl\Desktop\dataverse_files\HAM10000_images_part_1.zip"
zip2 = r"C:\Users\hgspl\Desktop\dataverse_files\HAM10000_images_part_2.zip"

output_dir = r"data\images"

os.makedirs(output_dir, exist_ok=True)


# -----------------------------
# 2. Read selected image IDs
# -----------------------------

df = pd.read_csv(csv_path)

selected_ids = set(df["image_id"].astype(str))

print("Selected images:", len(selected_ids))


# -----------------------------
# 3. Extract selected images
# -----------------------------

found = set()

for zip_path in [zip1, zip2]:

    print("\nChecking:")
    print(zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:

        for filename in z.namelist():

            image_name = os.path.basename(filename)

            if image_name.lower().endswith(".jpg"):

                image_id = os.path.splitext(image_name)[0]

                if image_id in selected_ids:

                    destination = os.path.join(output_dir, image_name)

                    if not os.path.exists(destination):
                        z.extract(filename, output_dir)

                        # Handle possible folder structure inside ZIP
                        extracted_path = os.path.join(
                            output_dir, filename
                        )

                        if os.path.exists(extracted_path):
                            os.replace(
                                extracted_path,
                                destination
                            )

                    found.add(image_id)

                    print(
                        f"Found {len(found)}/{len(selected_ids)}",
                        end="\r"
                    )


# -----------------------------
# 4. Check results
# -----------------------------

missing = selected_ids - found

print("\n\nFinished!")
print("Selected:", len(selected_ids))
print("Found:", len(found))
print("Missing:", len(missing))

if missing:
    print("\nMissing image IDs:")
    print(sorted(missing))

else:
    print("\nSUCCESS! All selected images were extracted.")