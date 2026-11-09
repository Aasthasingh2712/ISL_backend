import os

VIDEO_ROOT = r"dataset_raw\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Videos_Sentence_Level"

sentence_counts = {}

for folder_name in os.listdir(VIDEO_ROOT):
    folder_path = os.path.join(VIDEO_ROOT, folder_name)
    if os.path.isdir(folder_path):
        video_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".mp4", ".mov", ".avi"))]
        sentence_counts[folder_name] = len(video_files)

# Sort by video count, highest first
sorted_sentences = sorted(sentence_counts.items(), key=lambda x: x[1], reverse=True)

print(f"Total sentences: {len(sorted_sentences)}\n")
print(f"{'Sentence':<40} {'Video Count'}")
print("-" * 55)
for sentence, count in sorted_sentences:
    print(f"{sentence:<40} {count}")

print(f"\n--- Top 25 sentences (recommended for training) ---")
for sentence, count in sorted_sentences[:25]:
    print(f'"{sentence}",')