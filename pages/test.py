import json
import random
import streamlit as st
import sys
import os
import requests
from database import create_tables, get_latest_job_description
import sqlite3
from datetime import datetime
import time
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Groq API configuration
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def load_questions(file_path='questions.json'):
    """Load questions from a JSON file."""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return json.load(file)
        return {}
    except Exception as e:
        st.error(f"Error loading questions: {str(e)}")
        return {}

def fetch_questions_from_groq(job_description):
    """Fetch technical questions from Groq API based on job description."""
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        system_prompt = """You are a technical interviewer creating multiple choice questions. 
        Generate technical questions specific to the provided job description. Each question must be challenging 
        and relevant to assess the candidate's expertise for this specific role."""
        
        user_prompt = f"""Based on this job description:
        {job_description}
        
        Create 10 technical multiple choice questions that specifically test the skills and knowledge required for this role.
        Each question must follow this exact format:
        {{
            "question_text": "Technical question here",
            "options": ["Correct answer", "Wrong answer 1", "Wrong answer 2", "Wrong answer 3"],
            "correct_answer": 0,
            "explanation": "Explanation why the correct answer is right"
        }}
        
        Requirements:
        1. Questions must be technical and directly related to the job description
        2. Each question must have exactly 4 options
        3. The first option (index 0) must always be the correct answer
        4. Include clear explanations
        5. Format as a JSON array
        
        Return ONLY the JSON array with no additional text or formatting."""
        
        data = {
            "model": "mixtral-8x7b-32768",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 4000
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and result['choices']:
                content = result['choices'][0]['message']['content']
                
                # Clean up the response
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                # Parse questions
                questions = json.loads(content)
                if isinstance(questions, dict) and 'questions' in questions:
                    questions = questions['questions']
                elif not isinstance(questions, list):
                    questions = [questions]
                
                # Validate questions
                valid_questions = []
                for q in questions:
                    if (isinstance(q, dict) and 
                        'question_text' in q and 
                        'options' in q and 
                        'correct_answer' in q and 
                        'explanation' in q and 
                        isinstance(q['options'], list) and 
                        len(q['options']) == 4):
                        valid_questions.append(q)
                
                if valid_questions:
                    return valid_questions
        
        st.error(f"Failed to fetch questions from Groq API. Status: {response.status_code}")
        return None
        
    except Exception as e:
        st.error(f"Error fetching questions from Groq API: {str(e)}")
        return None

def update_questions_json(questions_dict, file_path='questions.json'):
    """Update questions.json with new questions."""
    try:
        with open(file_path, 'w') as f:
            json.dump(questions_dict, f, indent=4)
        return True
    except Exception as e:
        st.error(f"Error updating questions.json: {str(e)}")
        return False

def get_or_create_questions(job_description):
    """Get existing questions or fetch new ones from Groq API."""
    try:
        questions = load_questions()
        
        # Use job description as key
        description_key = job_description[:50]  # Use first 50 chars as key
        
        if not questions or description_key not in questions:
            new_questions = fetch_questions_from_groq(job_description)
            
            if new_questions:
                if not questions:
                    questions = {}
                questions[description_key] = new_questions
                if update_questions_json(questions):
                    return questions
                else:
                    st.error("Failed to save questions to file")
                    return None
            else:
                st.error("Failed to fetch questions from Groq API")
                return None
        
        return questions
    except Exception as e:
        st.error(f"Error in get_or_create_questions: {str(e)}")
        return None

def get_job_description_from_database():
    """Retrieve job description from database."""
    job_description = get_latest_job_description()
    
    if not job_description:
        st.error("No job description found. Please enter a job description first.")
        return None
        
    return job_description

def match_job_role(job_description):
    """Match job description to the closest job role category."""
    if not job_description:
        return 'software_developer'  # Default role if no description provided
        
    job_role_keywords = {
        'python_developer': [
            'python', 'django', 'flask', 'fastapi', 'web development', 'backend',
            'pandas', 'numpy', 'scikit-learn', 'pytest', 'pip', 'virtualenv'
        ],
        'java_developer': [
            'java', 'spring', 'hibernate', 'j2ee', 'backend', 'maven', 'gradle',
            'junit', 'servlets', 'jsp', 'microservices', 'spring boot'
        ],
        'web_development': [
            'frontend', 'html', 'css', 'javascript', 'web', 'responsive', 'ui/ux',
            'sass', 'less', 'bootstrap', 'tailwind', 'web design', 'spa'
        ],
        'data_science': [
            'data science', 'machine learning', 'python', 'statistics', 'analytics',
            'ai', 'artificial intelligence', 'deep learning', 'neural networks',
            'tensorflow', 'pytorch', 'data mining', 'nlp', 'computer vision'
        ],
        'database': [
            'sql', 'database', 'mysql', 'postgresql', 'oracle', 'mongodb',
            'nosql', 'redis', 'elasticsearch', 'data modeling', 'etl'
        ],
        'data_analyst': [
            'data analyst', 'data analysis', 'excel', 'sql', 'reporting',
            'business intelligence', 'data visualization', 'tableau', 'power bi',
            'statistics', 'analytics', 'metrics', 'kpi'
        ],
        'android_developer': [
            'android', 'kotlin', 'java', 'mobile development', 'android studio',
            'xml', 'gradle', 'app development', 'mobile apps', 'firebase',
            'material design', 'room database', 'jetpack compose'
        ],
        'cyber_security_specialist': [
            'security', 'cybersecurity', 'encryption', 'firewall', 'penetration testing',
            'vulnerability', 'network security', 'security audit', 'incident response',
            'malware', 'ethical hacking', 'siem', 'threat analysis'
        ],
        'software_developer': [
            'software development', 'programming', 'algorithms', 'git', 'api',
            'rest', 'unit testing', 'debugging', 'code review', 'agile',
            'software engineer', 'full stack', 'development'
        ],
        'front_end_developer': [
            'frontend', 'react', 'angular', 'vue', 'javascript', 'html', 'css',
            'responsive design', 'web development', 'ui/ux', 'typescript',
            'webpack', 'babel', 'redux', 'next.js'
        ],
        'database_administrator': [
            'database administration', 'sql', 'mysql', 'postgresql', 'oracle',
            'database security', 'backup', 'recovery', 'performance tuning',
            'data modeling', 'replication', 'clustering'
        ],
        'mobile_application_developer': [
            'mobile development', 'ios', 'android', 'swift', 'kotlin',
            'react native', 'flutter', 'mobile ui', 'app development',
            'mobile apps', 'xamarin', 'ionic', 'cordova'
        ],
        'network_architect': [
            'network architecture', 'cisco', 'routing', 'switching', 'tcp/ip',
            'vpn', 'wan', 'lan', 'network security', 'cloud networking',
            'sdwan', 'network design', 'infrastructure'
        ],
        'php_developer': [
            'php', 'laravel', 'symfony', 'wordpress', 'mysql', 'apache',
            'nginx', 'composer', 'php-fpm', 'backend', 'web development',
            'cms', 'drupal', 'codeigniter', 'lamp stack'
        ],
        'devops_engineer': [
            'devops', 'aws', 'azure', 'docker', 'kubernetes', 'jenkins',
            'ci/cd', 'terraform', 'ansible', 'cloud', 'automation',
            'infrastructure as code', 'monitoring'
        ],
        'full_stack_developer': [
            'full stack', 'frontend', 'backend', 'database', 'web development',
            'javascript', 'node.js', 'react', 'angular', 'vue', 'python',
            'java', 'php', 'full-stack', 'fullstack'
        ]
    }

    job_description = job_description.lower()
    role_scores = {}
    
    # Calculate scores for each role
    for role, keywords in job_role_keywords.items():
        # Base score from keyword matches
        keyword_score = sum(2 for keyword in keywords if keyword in job_description)
        
        # Additional score for exact phrase matches
        phrase_score = sum(3 for keyword in keywords if f" {keyword} " in f" {job_description} ")
        
        # Combine scores
        role_scores[role] = keyword_score + phrase_score
    
    if role_scores:
        max_score = max(role_scores.values())
        if max_score > 0:
            # Get the role with highest score
            matched_role = max(role_scores.items(), key=lambda x: x[1])[0]
            return matched_role
        
    # Default to software developer if no clear match
    return 'software_developer'
def cleanup_video_monitoring():
    """Clean up video monitoring resources"""
    if st.session_state.get('video_capture', None) is not None:
        st.session_state.video_capture.release()
        st.session_state.monitoring_active = False
        st.session_state.suspicious_behavior_count = 0
        st.session_state.violation_warned = False
        cv2.destroyAllWindows()

def get_shuffled_questions(questions, job_role, num_questions=10):
    """Get shuffled questions for the specified job role."""
    try:
        if not questions or job_role not in questions or not questions[job_role]:
            questions = get_or_create_questions(job_role)
            if not questions or job_role not in questions:
                st.error(f"Could not get questions for: {job_role}")
                return []
        
        job_role_questions = questions[job_role]
        if not job_role_questions:
            st.error("No questions available for this role")
            return []
        
        # Ensure we have enough questions
        if len(job_role_questions) < num_questions:
            st.warning(f"Only {len(job_role_questions)} questions available")
            return job_role_questions
        
        # Shuffle and return requested number of questions
        shuffled = job_role_questions.copy()
        random.shuffle(shuffled)
        return shuffled[:num_questions]
    except Exception as e:
        st.error(f"Error in get_shuffled_questions: {str(e)}")
        return []

def init_test_session_state():
    """Initialize test-related session state variables"""
    if 'current_test' not in st.session_state:
        params = st.query_params
        app_id = params.get("id")
        
        if app_id:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            try:
                c.execute('SELECT id FROM tests WHERE application_id = ?', (app_id,))
                test = c.fetchone()
                
                st.session_state.current_test = {
                    'test_id': test[0] if test else 'new',
                    'application_id': app_id
                }
            except Exception as e:
                st.error(f"Error initializing test: {str(e)}")
            finally:
                conn.close()
        else:
            st.error("No test ID provided")
            st.stop()
    
    # Initialize question navigation
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'responses' not in st.session_state:
        st.session_state.responses = {}
    
    # Initialize timer-related session state
    if 'start_time' not in st.session_state:
        st.session_state.start_time = time.time()
    if 'total_time' not in st.session_state:
        st.session_state.total_time = 600  # 10 minutes in seconds
    if 'test_submitted' not in st.session_state:
        st.session_state.test_submitted = False
    if 'show_details' not in st.session_state:
        st.session_state.show_details = False

def display_timer():
    """Display countdown timer in corner"""
    elapsed_time = int(time.time() - st.session_state.start_time)
    remaining_time = st.session_state.total_time - elapsed_time
    
    if remaining_time <= 0:
        st.error("Time's up! Submitting test...")
        submit_test()
        return False
    
    minutes = remaining_time // 60
    seconds = remaining_time % 60
    
    # Timer in corner with custom HTML/CSS
    st.markdown(f"""
        <div style='position: fixed; top: 70px; right: 20px; 
                    background-color: black; padding: 10px; 
                    border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    z-index: 1000;'>
            <p style='margin: 0; font-size: 14px; color: #666;'>Time Remaining</p>
            <p style='margin: 0; font-size: 20px; font-weight: bold; 
                      color: {"red" if remaining_time < 60 else "#333"};'>
                {minutes:02d}:{seconds:02d}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    return True

def init_video_monitoring():
    """Initialize YOLO model and video monitoring components"""
    if 'model' not in st.session_state:
        st.session_state.model = YOLO('yolov8n.pt')  # Load YOLO model
    if 'video_frame' not in st.session_state:
        st.session_state.video_frame = None
    if 'monitoring_active' not in st.session_state:
        st.session_state.monitoring_active = False
    if 'suspicious_behavior_count' not in st.session_state:
        st.session_state.suspicious_behavior_count = 0
    if 'total_violations' not in st.session_state:
        st.session_state.total_violations = 0
    if 'violation_warned' not in st.session_state:
        st.session_state.violation_warned = False

def process_frame(frame):
    """Process video frame with YOLO model"""
    try:
        # Run YOLO detection
        results = st.session_state.model(frame)
        
        # Process results
        for result in results:
            boxes = result.boxes
            suspicious_count = 0
            person_count = 0
            
            for box in boxes:
                # Get class name
                class_name = result.names[int(box.cls[0])]
                
                # Count persons and draw blue border around faces
                if class_name == 'person':
                    person_count += 1
                    x1, y1, x2, y2 = box.xyxy[0]
                    cv2.rectangle(frame, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                (255, 0, 0), 2)  # Blue (BGR format)
                
                # Check for suspicious behavior
                if class_name in ['cell phone', 'book']:
                    st.session_state.suspicious_behavior_count += 1
                    suspicious_count += 1
                    
                    # Draw bounding box in blue
                    x1, y1, x2, y2 = box.xyxy[0]
                    cv2.rectangle(frame, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                (255, 0, 0), 2)  # Blue (BGR format)
                    
                    # Add label for suspicious items in red
                    cv2.putText(frame, 
                              f"{class_name} detected!", 
                              (int(x1), int(y1)-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.9, 
                              (0, 0, 255),  # Red (BGR format)
                              2)
            
            # Check for multiple faces or no face
            if person_count > 1:
                # Draw red border around entire frame for alert
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 3)  # Red border
                
                cv2.putText(frame,
                           "Multiple faces detected!",
                           (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           1,
                           (0, 0, 255),  # Red color
                           2)
                suspicious_count += 1
                st.session_state.suspicious_behavior_count += 1
            elif person_count == 0:
                cv2.putText(frame,
                           "No face detected!",
                           (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           1,
                           (0, 0, 255),  # Red color
                           2)
                suspicious_count += 1
                st.session_state.suspicious_behavior_count += 1
            
            # Update total violations in session state
            if 'total_violations' not in st.session_state:
                st.session_state.total_violations = 0
            st.session_state.total_violations += suspicious_count
        
        return frame
    
    except Exception as e:
        st.error(f"Error processing frame: {str(e)}")
        return frame

def update_video_feed():
    """Update video feed with YOLO detection"""
    if st.session_state.get('monitoring_active', False):
        try:
            ret, frame = st.session_state.video_capture.read()
            if ret:
                # Process frame with YOLO
                processed_frame = process_frame(frame)
                
                # Create placeholders if they don't exist
                if 'video_placeholder' not in st.session_state:
                    st.session_state.video_placeholder = st.empty()
                if 'warning_placeholder' not in st.session_state:
                    st.session_state.warning_placeholder = st.empty()
                if 'warning_count_placeholder' not in st.session_state:
                    st.session_state.warning_count_placeholder = st.empty()
                
                # Display the processed frame
                st.session_state.video_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
                
                # Display warnings
                if st.session_state.suspicious_behavior_count > 0:
                    st.session_state.warning_placeholder.warning("Warning: Suspicious behavior detected!")
                    st.session_state.warning_count_placeholder.info(
                        f"Total Violations Detected: {st.session_state.suspicious_behavior_count}"
                    )
                
                # Check violation threshold
                if (st.session_state.suspicious_behavior_count >= 5 and 
                    not st.session_state.violation_warned):
                    st.error("Multiple violations detected! Test session has been flagged.")
                    st.session_state.violation_warned = True
                
                # Store current frame
                st.session_state.video_frame = processed_frame
                
                return True
            
        except Exception as e:
            st.error(f"Error updating video feed: {str(e)}")
            return False
    
    return False

def setup_video_monitoring():
    """Setup video monitoring with YOLO detection"""
    # Initialize video capture
    if not st.session_state.get('monitoring_active', False):
        try:
            video_capture = cv2.VideoCapture(0)
            if video_capture.isOpened():
                st.session_state.video_capture = video_capture
                st.session_state.monitoring_active = True
                init_video_monitoring()
            else:
                st.error("Could not access webcam")
                return False
        except Exception as e:
            st.error(f"Error initializing video capture: {str(e)}")
            return False
    
    return True

def cleanup_video_monitoring():
    """Clean up video monitoring resources"""
    if st.session_state.get('video_capture', None) is not None:
        st.session_state.video_capture.release()
        st.session_state.monitoring_active = False
        st.session_state.suspicious_behavior_count = 0
        st.session_state.violation_warned = False
        cv2.destroyAllWindows()

def display_question_form(questions):
    """Display questions with video monitoring"""
    if not setup_video_monitoring():
        st.error("Video monitoring is required to take the test")
        return
    
    # Start the timer
    if not display_timer():
        cleanup_video_monitoring()
        return
    
    current_q = st.session_state.current_question
    total_questions = len(questions)
    
    # Create two columns: one for questions and one for video
    col1, col2 = st.columns([3, 1])
    
    # Main content container
    with col1:
        # Progress bar at the top
        progress_text = f"Question {current_q + 1} of {total_questions}"
        st.progress((current_q + 1) / total_questions, text=progress_text)
        
        # Question display
        st.markdown("---")
        question = questions[current_q]
        
        # Question container
        st.markdown(f"""
            <div style='margin-top: 20px;'>
                <h3>Question {current_q + 1}</h3>
                <p style='font-size: 18px;'>{question['question_text']}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Options
        selected_answer = st.radio(
            "Select your answer:",
            options=question['options'],
            key=f"q_{current_q}",
            index=None
        )
        
        # Store response
        if selected_answer:
            st.session_state.responses[f"q_{current_q}"] = {
                'question': question['question_text'],
                'selected_answer': selected_answer,
                'correct_answer': question['options'][question['correct_answer']]
            }
        
        st.markdown("---")
        
        # Navigation buttons
        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
        
        with nav_col1:
            if current_q > 0:
                if st.button("← Previous", use_container_width=True):
                    st.session_state.current_question -= 1
                    st.rerun()
        
        with nav_col2:
            if current_q == total_questions - 1 and selected_answer:
                if st.button("Submit Test", type="primary", use_container_width=True):
                    submit_test()
        
        with nav_col3:
            if current_q < total_questions - 1:
                if st.button("Next →", use_container_width=True):
                    if selected_answer:
                        st.session_state.current_question += 1
                        st.rerun()
                    else:
                        st.warning("Please select an answer before proceeding.")
        
        # Progress indicator
        st.markdown(f"""
            <div style='text-align: center; margin-top: 20px;'>
                <p>Progress: {len(st.session_state.responses)}/{total_questions} questions answered</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Video monitoring column
    with col2:
        st.markdown("""
            <div style="position: sticky; top: 80px;">
                <h4 style="margin-bottom: 10px;">📹 Video Monitoring</h4>
                <p style="font-size: 0.8em; color: #666; margin-bottom: 10px;">
                    Please ensure:
                    • Only one person visible
                    • No phones or devices
                    • Stay in camera view
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Update video feed
        update_video_feed()
    
    # Auto-refresh timer
    st.empty()
    time.sleep(0.1)  # Small delay to prevent overwhelming the system
    st.rerun()

def calculate_score(responses, questions):
    """Calculate the score based on correct answers."""
    score = 0
    total_questions = len(questions)
    
    for i in range(total_questions):
        q_key = f"q_{i}"
        if q_key in responses:
            response = responses[q_key]
            question = questions[i]
            if response['selected_answer'] == question['options'][question['correct_answer']]:
                score += 1
    
    return score / total_questions  # Return score as a fraction of 1

def submit_test():
    """Handle test submission and cleanup"""
    # Turn off camera and release resources
    if st.session_state.get('video_capture', None) is not None:
        st.session_state.video_capture.release()
        st.session_state.monitoring_active = False
        cv2.destroyAllWindows()
    
    try:
        # Calculate score
        score = calculate_score(st.session_state.responses, st.session_state.questions)
        
        # Get total violations from session state
        total_violations = st.session_state.get('total_violations', 0)
        
        # Update database
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        try:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # First check if test record exists
            c.execute('''
                SELECT id FROM tests 
                WHERE application_id = ?
            ''', (st.session_state.current_test['application_id'],))
            
            test_record = c.fetchone()
            
            if not test_record:
                # Insert new test record
                c.execute('''
                    INSERT INTO tests 
                    (application_id, status, score, completed_at, violations_count)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    st.session_state.current_test['application_id'],
                    'completed',
                    score * 100,
                    current_time,
                    total_violations
                ))
                test_id = c.lastrowid
            else:
                # Update existing test record
                test_id = test_record[0]
                c.execute('''
                    UPDATE tests 
                    SET status = ?,
                        score = ?,
                        completed_at = ?,
                        violations_count = ?
                    WHERE id = ?
                ''', (
                    'completed',
                    score * 100,
                    current_time,
                    total_violations,
                    test_id
                ))
            
            # Save test results
            c.execute('''
                INSERT INTO test_results 
                (test_id, answers, score, completed_at, violations_count)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                test_id,
                json.dumps(st.session_state.responses),
                score * 100,
                current_time,
                total_violations
            ))
            
            conn.commit()
            
            # Set submission state
            st.session_state.test_submitted = True
            st.session_state.final_score = score
            st.session_state.final_violations = total_violations
            st.rerun()
            
        except Exception as e:
            st.error(f"Error updating test status: {str(e)}")
            conn.rollback()
        finally:
            conn.close()
            
    except Exception as e:
        st.error(f"Error submitting test: {str(e)}")

def display_results():
    """Display test results after submission"""
    if not hasattr(st.session_state, 'test_submitted'):
        return
    
    score = st.session_state.final_score
    violations = st.session_state.final_violations
    
    st.success("🎉 Test Completed Successfully!")
    
    # Create a container for the report card
    with st.container():
        st.markdown("""
            <div style='background-color: #000000; padding: 20px; border-radius: 10px; margin: 20px 0;'>
                <h2 style='text-align: center; color: #ffffff; margin-bottom: 20px;'>Assessment Report</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Create columns for score and violations
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Final Score",
                f"{score * 100:.2f}%",
                f"{int(score * 10)} out of 10 correct",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                "Total Violations Detected",
                f"{violations}",
                "During test session",
                delta_color="inverse" if violations > 0 else "off"
            )
        
        # Violation details section
        if violations > 0:
            st.markdown("""
                <div style='background-color: #000000; padding: 20px; border-radius: 10px; margin: 20px 0;'>
                    <h4 style='color: #ffffff; margin-bottom: 15px;'>⚠️ Violation Details</h4>
                    <div style='background-color: black; padding: 15px; border-radius: 8px;'>
                        <p style='margin: 0; color: #666;'><strong>Violations Breakdown:</strong></p>
                        <ul style='margin: 10px 0;'>
                            <li>Multiple people detected in frame</li>
                            <li>Looking away from screen frequently</li>
                            <li>Unauthorized devices or materials detected</li>
                        </ul>
                        <p style='margin: 10px 0 0 0; color: #dc3545;'>
                            These violations may impact your final evaluation. Please ensure proper test conditions in future attempts.
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Display different messages based on score and violations
        if score >= 0.8 and violations <= 2:
            st.success("🌟 Excellent performance with good test conduct!")
        elif score >= 0.6 and violations <= 5:
            st.warning("📝 Good performance but some monitoring concerns noted.")
        else:
            st.error("⚠️ Performance needs improvement or test conduct was questionable.")
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("View Detailed Results", key="view_details"):
                st.session_state.show_details = True
                st.rerun()
        
        with col2:
            if st.button("Return to Profile", type="primary", key="return_profile"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.switch_page("app.py")
    
    # Detailed results section
    if st.session_state.get('show_details', False):
        st.markdown("### Detailed Results")
        
        for i, (q_key, response) in enumerate(st.session_state.responses.items()):
            with st.expander(f"Question {i + 1}"):
                correct = response['selected_answer'] == response['correct_answer']
                st.markdown(f"""
                    <div style='padding: 10px;'>
                        <p><strong>Question:</strong> {response['question']}</p>
                        <p><strong>Your Answer:</strong> {response['selected_answer']}</p>
                        <p><strong>Correct Answer:</strong> {response['correct_answer']}</p>
                        <p style='color: {"#2e7d32" if correct else "#d32f2f"}'>
                            <strong>Status:</strong> {"✅ Correct" if correct else "❌ Incorrect"}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

def update_candidate_status(candidate_id, score):
    """Update the candidate's assessment status and score in the database."""
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        # Update candidate status and score
        c.execute('''
            UPDATE candidates
            SET assessment_status = 'Test Completed',
                test_score = ?,
                completion_date = datetime('now')
            WHERE id = ?
        ''', (score, candidate_id))
        
        conn.commit()
        
        # Fetch updated candidate details to display
        c.execute('''
            SELECT assessment_status, test_score, completion_date 
            FROM candidates 
            WHERE id = ?
        ''', (candidate_id,))
        
        status_details = c.fetchone()
        if status_details:
            status, test_score, completion_date = status_details
            
            # Display updated status in a formatted box
            st.markdown("""
            <div style='padding: 20px; border-radius: 10px; background-color: #f0f2f6; margin: 20px 0;'>
                <h3 style='color: #0066cc;'>Assessment Status</h3>
                <hr>
                <p><strong>Status:</strong> {}</p>
                <p><strong>Score:</strong> {:.2f}%</p>
                <p><strong>Completed On:</strong> {}</p>
            </div>
            """.format(status, test_score, completion_date), unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error updating status: {str(e)}")
    finally:
        conn.close()

def get_test_analysis(test_id):
    """Get detailed test analysis including violations and score"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        # Get test details
        c.execute('''
            SELECT t.*, tr.answers, tr.completed_at,
                   a.user_id, u.username
            FROM tests t
            JOIN test_results tr ON t.id = tr.test_id
            JOIN applications a ON t.application_id = a.id
            JOIN users u ON a.user_id = u.id
            WHERE t.id = ?
        ''', (test_id,))
        
        test_data = c.fetchone()
        if test_data:
            return {
                'test_id': test_data[0],
                'score': test_data[3],
                'answers': json.loads(test_data[5]),
                'completion_time': test_data[6],
                'user_id': test_data[7],
                'username': test_data[8],
                'violations': st.session_state.get('warnings', {
                    'no_person': 0,
                    'multiple_people': 0,
                    'device_detected': 0
                })
            }
    except Exception as e:
        st.error(f"Error fetching test analysis: {str(e)}")
    finally:
        conn.close()
    return None

def display_test_analysis(test_data):
    """Display test analysis in HR dashboard"""
    st.markdown("### Test Analysis")
    
    # Create three columns for key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Final Score", f"{test_data['score']}%")
    
    with col2:
        total_violations = sum(test_data['violations'].values())
        st.metric("Total Violations", total_violations)
    
    with col3:
        completion_time = datetime.strptime(test_data['completion_time'], '%Y-%m-%d %H:%M:%S')
        st.metric("Completion Time", completion_time.strftime('%Y-%m-%d %H:%M'))
    
    # Violation breakdown
    st.markdown("#### Violation Breakdown")
    violations = test_data['violations']
    
    # Create a bar chart for violations
    fig = go.Figure(data=[
        go.Bar(
            x=list(violations.keys()),
            y=list(violations.values()),
            marker_color=['#FF9999', '#FFB366', '#99FF99']
        )
    ])
    
    fig.update_layout(
        title="Violation Types",
        xaxis_title="Violation Type",
        yaxis_title="Count",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Answer analysis
    st.markdown("#### Answer Analysis")
    for q_num, (q_key, response) in enumerate(test_data['answers'].items()):
        with st.expander(f"Question {q_num}"):
            correct = response['selected_answer'] == response['correct_answer']
            st.markdown(f"""
                **Question:** {response['question']}  
                **Selected Answer:** {response['selected_answer']}  
                **Correct Answer:** {response['correct_answer']}  
                **Status:** {'✅ Correct' if correct else '❌ Incorrect'}
            """)
    
    # Add recommendations based on performance
    st.markdown("#### Recommendations")
    score = test_data['score']
    violations = sum(test_data['violations'].values())
    
    if score >= 80 and violations <= 2:
        st.success("Strong candidate with good test conduct")
    elif score >= 60 and violations <= 5:
        st.warning("Moderate performance with some monitoring concerns")
    else:
        st.error("Performance needs improvement or test conduct was questionable")
    
    # Add detailed violation analysis if any
    if violations > 0:
        st.markdown("#### Violation Details")
        for violation_type, count in violations.items():
            if count > 0:
                st.warning(f"{violation_type.replace('_', ' ').title()}: {count} occurrences")

def render_hr_dashboard():
    """Render HR dashboard with test analysis"""
    st.title("HR Dashboard")
    
    # Add tabs for different views
    tab1, tab2, tab3 = st.tabs(["Applications", "Test Results", "Candidate Profiles"])
    
    with tab1:
        render_hr_applications()
    
    with tab2:
        st.subheader("Test Results Analysis")
        # Get all completed tests
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('''
            SELECT t.id, u.username, t.score, t.completed_at
            FROM tests t
            JOIN applications a ON t.application_id = a.id
            JOIN users u ON a.user_id = u.id
            WHERE t.status = 'completed'
            ORDER BY t.completed_at DESC
        ''')
        tests = c.fetchall()
        conn.close()
        
        if tests:
            for test in tests:
                with st.expander(f"Test Result - {test[1]} ({test[3]})"):
                    test_data = get_test_analysis(test[0])
                    if test_data:
                        display_test_analysis(test_data)
        else:
            st.info("No completed tests found")
    
    with tab3:
        if 'viewing_candidate' in st.session_state:
            if st.button("← Back to Candidate List"):
                del st.session_state.viewing_candidate
                st.rerun()
            render_candidate_view(st.session_state.viewing_candidate, "hr")
        else:
            render_candidate_list()

def update_database_schema():
    """Update database schema to include violations_count"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    try:
        # Check if violations_count column exists in tests table
        c.execute("PRAGMA table_info(tests)")
        columns = [col[1] for col in c.fetchall()]
        
        # Add violations_count column if it doesn't exist
        if 'violations_count' not in columns:
            c.execute('ALTER TABLE tests ADD COLUMN violations_count INTEGER DEFAULT 0')
        
        # Check if violations_count column exists in test_results table
        c.execute("PRAGMA table_info(test_results)")
        columns = [col[1] for col in c.fetchall()]
        
        # Add violations_count column if it doesn't exist
        if 'violations_count' not in columns:
            c.execute('ALTER TABLE test_results ADD COLUMN violations_count INTEGER DEFAULT 0')
        
        conn.commit()
        print("Database schema updated successfully!")
        
    except Exception as e:
        print(f"Error updating database schema: {str(e)}")
        conn.rollback()
    finally:
        conn.close()

def create_tables():
    """Create necessary database tables"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    try:
        # Create tests table with violations_count
        c.execute('''
            CREATE TABLE IF NOT EXISTS tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                application_id INTEGER,
                status TEXT DEFAULT 'pending',
                score REAL,
                completed_at DATETIME,
                violations_count INTEGER DEFAULT 0,
                FOREIGN KEY (application_id) REFERENCES applications (id)
            )
        ''')
        
        # Create test_results table with violations_count
        c.execute('''
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id INTEGER,
                answers TEXT,
                score REAL,
                completed_at DATETIME,
                violations_count INTEGER DEFAULT 0,
                FOREIGN KEY (test_id) REFERENCES tests (id)
            )
        ''')
        
        conn.commit()
        print("Tables created successfully!")
        
    except Exception as e:
        print(f"Error creating tables: {str(e)}")
        conn.rollback()
    finally:
        conn.close()

def main():
    # Create tables and update schema
    create_tables()
    update_database_schema()
    
    st.title("Technical Skills Assessment")
    
    # Initialize session state
    init_test_session_state()
    
    if st.session_state.get('test_submitted', False):
        display_results()
        return
    
    # Get job description first
    job_description = get_job_description_from_database()
    if job_description is None:
        st.error("Please provide a job description before starting the assessment.")
        return
    
    # Load or fetch questions directly from job description
    if 'questions' not in st.session_state:
        questions = load_questions()
        description_key = job_description[:50]  # Use first 50 chars as key
        
        if not questions or description_key not in questions:
            questions = get_or_create_questions(job_description)
            if not questions or description_key not in questions:
                st.error("Failed to fetch questions. Please try again.")
                return
    
    # Instructions and start button
    if 'test_started' not in st.session_state:
        st.markdown("""
            <div style='padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                <h3>📝 Instructions</h3>
                <ul>
                    <li>Duration: 10 minutes</li>
                    <li>10 multiple choice questions</li>
                    <li>Navigate between questions using Previous/Next buttons</li>
                    <li>Submit before time runs out</li>
                    <li>Timer starts when you begin the test</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Begin Assessment", key="start_assessment", use_container_width=True):
            # Get shuffled questions when starting the test
            description_key = job_description[:50]
            shuffled_questions = get_shuffled_questions(questions, description_key)
            if shuffled_questions:
                st.session_state.test_started = True
                st.session_state.questions = shuffled_questions
                st.rerun()
            else:
                st.error("Could not prepare questions. Please try again.")
    
    # Display test if started
    elif st.session_state.get('test_started', False):
        if st.session_state.get('questions'):
            display_question_form(st.session_state.questions)
        else:
            st.error("No questions loaded. Please restart the assessment.")
            st.warning("No questions available for this job description.")

if __name__ == "__main__":
    main()
