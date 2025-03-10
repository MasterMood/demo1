import sqlite3
import json

def create_question_database():
    conn = sqlite3.connect('question_database.db')
    c = conn.cursor()
    
    # Create questions table
    c.execute('''CREATE TABLE IF NOT EXISTS questions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  skill TEXT,
                  question TEXT,
                  options TEXT,
                  correct_answer TEXT,
                  explanation TEXT)''')
    
    # Sample questions dataset
    questions = [
        # Java Questions
        {
            'skill': 'java',
            'question': 'What is inheritance in Java?',
            'options': json.dumps([
                'Process of creating multiple objects',
                'Mechanism where one class acquires properties of another class',
                'Method of creating variables',
                'None of the above'
            ]),
            'correct_answer': 'Mechanism where one class acquires properties of another class',
            'explanation': 'Inheritance is a mechanism that allows a class to inherit properties and methods from another class.'
        },
        {
            'skill': 'java',
            'question': 'Which of these keywords is used to define interfaces in Java?',
            'options': json.dumps([
                'interface',
                'Interface',
                'intf',
                'Inherit'
            ]),
            'correct_answer': 'interface',
            'explanation': 'The interface keyword is used to declare an interface in Java.'
        },
        
        # Android Questions
        {
            'skill': 'android',
            'question': 'What is an Activity in Android?',
            'options': json.dumps([
                'A single screen in an app',
                'A database operation',
                'A background service',
                'A type of layout'
            ]),
            'correct_answer': 'A single screen in an app',
            'explanation': 'An Activity represents a single screen with a user interface in Android applications.'
        },
        {
            'skill': 'android',
            'question': 'What is the purpose of Intent in Android?',
            'options': json.dumps([
                'To store data',
                'To perform calculations',
                'To communicate between components',
                'To design UI'
            ]),
            'correct_answer': 'To communicate between components',
            'explanation': 'Intent is used to communicate between components and navigate between different screens in Android.'
        }
        # Add more questions as needed
    ]
    
    # Insert questions into database
    for q in questions:
        c.execute('''
            INSERT OR IGNORE INTO questions
            (skill, question, options, correct_answer, explanation)
            VALUES (?, ?, ?, ?, ?)
        ''', (q['skill'], q['question'], q['options'], q['correct_answer'], q['explanation']))
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_question_database()
