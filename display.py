import matplotlib.pyplot as plt

def overlap(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return x1 < x2 and y1 < y2

with open("output.txt", "r") as file:
    data = file.read().splitlines()

time_series = []
persons = []
others = []
has_overlap = 0
current_time = 0
stage_duration = 5

for item in data:
    parts = item.split(',')
    if len(parts) == 6:
        item_time = float(parts[0])
        
        if item_time >= current_time + stage_duration:
            if has_overlap:
                time_series.append(1)
            else:
                time_series.append(0)
            persons = []
            others = []
            has_overlap = 0
            current_time = item_time - (item_time % stage_duration)

        obj_type = parts[1]
        coords = list(map(int, parts[2:]))
        if obj_type == "person":
            for other in others:
                if overlap(coords, other):
                    has_overlap = 1
                    break
            if not has_overlap:
                persons.append(coords)
        elif obj_type in ["mouse", "keyboard"]:
            for person in persons:
                if overlap(person, coords):
                    has_overlap = 1
                    break
            if not has_overlap:
                others.append(coords)


time_stamps = [i * stage_duration for i in range(len(time_series))]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.step(time_stamps, time_series, where='post')
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Stage')
ax1.set_title('Time Series Stage Plot')
ax1.set_yticks([0, 1])
ax1.set_yticklabels(['Not Learning', 'Learning'])
ax1.grid(True)

learning_count = time_series.count(1)
not_learning_count = time_series.count(0)
sizes = [not_learning_count, learning_count]
ax2.pie(sizes, labels=['Not Learning', 'Learning'], autopct='%1.1f%%', startangle=90)
ax2.set_title('Learning vs Not Learning Proportion')
ax2.axis('equal')

plt.savefig('behaviors.png')