import os
import pandas as pd
import random

class FeatureSelectionService:

    @staticmethod
    def pick_evenly_distributed_items(lst, num_items):
        if num_items > len(lst):
            raise ValueError("Number of items to pick cannot exceed the length of the list")
        
        # Calculate the size of each segment
        segment_size = len(lst) // num_items
        
        # Pick one item from each segment
        selected_items = []
        for i in range(num_items):
            # Determine the start of the current segment
            start = i * segment_size
            # If it's the last segment, it can go up to the end of the list
            end = start + segment_size if i < num_items - 1 else len(lst)
            # Pick a random item from the segment
            random_index = random.randint(start, end - 1)
            selected_items.append(lst[random_index])
        
        return selected_items
    
    @staticmethod
    def get_feature_candidates_to_remove(builds, k=5):
        candidates = {}
        for build in builds:
            df = pd.read_csv(os.path.join(build, 'feature_stats.csv'))
            sorted_df = df.sort_values(by='frequency', ascending=True)
            N = len(sorted_df)

            for index, feature_id in enumerate(sorted_df['feature_id'].tolist()):
                if feature_id in candidates:
                    candidates[feature_id] += (N - index) * (N - index)
                else:
                    candidates[feature_id] = (N - index) * (N - index)
        
        x = pd.DataFrame(list(candidates.items()), columns=['feature_id', 'rank'])
        return x.sort_values(by='rank', ascending=False).head(k)['feature_id'].tolist()
