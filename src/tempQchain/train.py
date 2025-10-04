"""
Dummy dataset with correct transitive relations in the same format as training_df.txt
Each batch is a dict with questions@@separated, and relation IDs match question_ids
"""

dummy_train_data = [
    # Batch 1: Simple transitive chain with 3 questions
    {
        'questions': 'When did e1 happen compared to e2?@@When did e2 happen compared to e3?@@When did e1 happen compared to e3?',
        'stories': 'Story text here...@@Story text here...@@Story text here...',
        'relation': '@@@@transitive,0,1',  # Empty for Q0, Empty for Q1, Q2 is transitive(Q0, Q1)
        'question_ids': '0@@1@@2',
        'labels': '1@@1@@1'  # All "before" (label 1)
    },
    
    # Batch 2: Another transitive chain with 4 questions
    {
        'questions': 'When did e4 happen compared to e5?@@When did e5 happen compared to e6?@@When did e4 happen compared to e6?@@When did e6 happen compared to e7?',
        'stories': 'Story text here...@@Story text here...@@Story text here...@@Story text here...',
        'relation': '@@@@transitive,0,1@@',  # Q2 is transitive(Q0, Q1)
        'question_ids': '0@@1@@2@@3',
        'labels': '2@@2@@2@@2'  # All "after" (label 2)
    },
    
    # Batch 3: Multiple transitive relations
    {
        'questions': 'When did e7 happen compared to e8?@@When did e8 happen compared to e9?@@When did e7 happen compared to e9?@@When did e9 happen compared to e10?@@When did e7 happen compared to e10?',
        'stories': 'Story text here...@@Story text here...@@Story text here...@@Story text here...@@Story text here...',
        'relation': '@@@@transitive,0,1@@@@transitive,2,3',  # Q2=trans(Q0,Q1), Q4=trans(Q2,Q3)
        'question_ids': '0@@1@@2@@3@@4',
        'labels': '1@@1@@1@@1@@1'  # All "before"
    },
    
    # Batch 4: Symmetric relation example
    {
        'questions': 'When did e11 happen compared to e12?@@When did e12 happen compared to e11?@@When did e11 happen compared to e13?',
        'stories': 'Story text here...@@Story text here...@@Story text here...',
        'relation': '@@symmetric,1@@',  # Q1 is symmetric to Q0
        'question_ids': '0@@1@@2',
        'labels': '32@@32@@1'  # simultaneous, simultaneous, before
    },
    
    # Batch 5: Complex chain with both transitive and symmetric
    {
        'questions': 'When did e14 happen compared to e15?@@When did e15 happen compared to e16?@@When did e14 happen compared to e16?@@When did e16 happen compared to e14?',
        'stories': 'Story text here...@@Story text here...@@Story text here...@@Story text here...',
        'relation': '@@@@transitive,0,1@@symmetric,3',  # Q2=trans(Q0,Q1), Q3=symmetric(Q2)
        'question_ids': '0@@1@@2@@3',
        'labels': '1@@1@@1@@2'  # before, before, before, after
    },
]

# Function to convert to the format expected by your program
def convert_to_tuples(batch_dict):
    """Convert dict format to tuple format if needed"""
    questions = batch_dict['questions'].split('@@')
    stories = batch_dict['stories'].split('@@')
    relations = batch_dict['relation'].split('@@')
    question_ids = batch_dict['question_ids'].split('@@')
    labels = batch_dict['labels'].split('@@')
    
    batch_tuples = []
    for i in range(len(questions)):
        batch_tuples.append((
            questions[i],
            stories[i],
            relations[i],
            question_ids[i],
            labels[i]
        ))
    return batch_tuples