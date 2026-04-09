from tasks import ALL_TASKS, grader

max_score = -1.0
min_score = 2.0

for t in ALL_TASKS:
    # Test perfect
    s_perf = grader(t, {"intent": t.ground_truth_intent, "priority": t.ground_truth_priority, "reply": "sorry we apologize log troubleshoot invoice credit", "submitted": True})
    # Test empty
    s_empty = grader(t, {})
    
    max_score = max(max_score, s_perf, s_empty)
    min_score = min(min_score, s_perf, s_empty)

print(f"Min: {min_score}, Max: {max_score}")
