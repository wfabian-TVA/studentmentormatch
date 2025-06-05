import streamlit as st
import pandas as pd
import numpy as np
import math
import base64
import random
from io import BytesIO

st.set_page_config(page_title="Speed Networking Matcher", layout="wide")

st.title("Speed Networking Matcher")
st.write("""
This app helps you match students with mentors for speed networking sessions.
On the righthand side, please choose the number of conversations you would like each student and mentor to have and the timing of the session.
""")

# Session state initialization
if 'schedule_generated' not in st.session_state:
    st.session_state.schedule_generated = False
if 'all_students_matched' not in st.session_state:
    st.session_state.all_students_matched = False

# Helper functions
def calculate_match_score(student_interests, mentor_interests):
    """Calculate matching score between a student and mentor based on interests"""
    # Convert interests to sets and find intersection
    # Filter out NaN values and convert to strings
    student_set = set(
        str(interest).strip().lower() 
        for interest in student_interests 
        if interest is not None and pd.notna(interest) and str(interest).strip()
    )
    mentor_set = set(
        str(interest).strip().lower() 
        for interest in mentor_interests 
        if interest is not None and pd.notna(interest) and str(interest).strip()
    )
    
    # Count matching interests
    matching = student_set.intersection(mentor_set)
    
    if not student_set or not mentor_set:
        return 0
    
    # Jaccard similarity: intersection / union
    score = len(matching) / math.sqrt(len(student_set) * len(mentor_set))
    return score

def can_assign_meeting(meeting, trackers, student_target, mentor_max):
    """Check if a meeting can be assigned given current constraints"""
    s_id = meeting['student_id']
    m_id = meeting['mentor_id']
    t_slot = meeting['time_slot']
    
    return (trackers['student_meeting_counts'][s_id] < student_target and  # Student needs more meetings
            trackers['mentor_meeting_counts'][m_id] < mentor_max and   # Mentor hasn't exceeded max
            t_slot not in trackers['student_schedule'][s_id] and  # Student is available
            t_slot not in trackers['mentor_schedule'][m_id] and   # Mentor is available
            m_id not in trackers['student_met_mentors'][s_id])   # Student hasn't met this mentor

def make_assignment(meeting, trackers):
    """Assign a meeting and update all tracking variables"""
    s_id = meeting['student_id']
    m_id = meeting['mentor_id']
    t_slot = meeting['time_slot']
    
    # Add the assignment
    trackers['assignments'].append(meeting)
    
    # Update trackers
    trackers['student_meeting_counts'][s_id] += 1
    trackers['mentor_meeting_counts'][m_id] += 1
    trackers['student_schedule'][s_id].append(t_slot)
    trackers['mentor_schedule'][m_id].append(t_slot)
    trackers['student_met_mentors'][s_id].append(m_id)

def assign_meetings(meetings_df, trackers, student_target, mentor_max):
    """Assign meetings from a sorted DataFrame"""
    for _, meeting in meetings_df.iterrows():
        if can_assign_meeting(meeting, trackers, student_target, mentor_max):
            make_assignment(meeting, trackers)

def generate_matches_enterprise(students_df, mentors_df, time_slots, student_meetings_target=3, mentor_meetings_max=5):
    """
    Advanced matching algorithm optimized for large-scale networking
    Uses multiple passes and prioritization to ensure all students get their target meetings
    """
    
    # Create compatibility matrix
    compatibility_scores = {}
    potential_meetings = []
    
    st.write("Building compatibility matrix...")
    progress_bar = st.progress(0)
    total_combinations = len(students_df) * len(mentors_df)
    current_combination = 0
    
    for s_idx, student in students_df.iterrows():
        student_interests = [student[col] for col in students_df.columns if 'interest' in col.lower()]
        
        for m_idx, mentor in mentors_df.iterrows():
            mentor_interests = [mentor[col] for col in mentors_df.columns if 'interest' in col.lower()]
            
            # Calculate match score
            score = calculate_match_score(student_interests, mentor_interests)
            compatibility_scores[(student['StudentID'], mentor['MentorID'])] = score
            
            # Add all possible time slot combinations
            for time_slot in time_slots:
                potential_meetings.append({
                    'student_id': student['StudentID'],
                    'student_name': student['Student Name'],
                    'mentor_id': mentor['MentorID'],
                    'mentor_name': mentor['Name'],
                    'time_slot': time_slot,
                    'score': score
                })
            
            current_combination += 1
            progress_bar.progress(current_combination / total_combinations)
    
    # Convert to DataFrame
    meetings_df = pd.DataFrame(potential_meetings)
    
    # Initialize tracking variables
    def reset_trackers():
        return {
            'assignments': [],
            'student_meeting_counts': {sid: 0 for sid in students_df['StudentID']},
            'mentor_meeting_counts': {mid: 0 for mid in mentors_df['MentorID']},
            'student_schedule': {sid: [] for sid in students_df['StudentID']},
            'mentor_schedule': {mid: [] for mid in mentors_df['MentorID']},
            'student_met_mentors': {sid: [] for sid in students_df['StudentID']}
        }
    
    best_result = None
    best_completed_students = 0
    
    # Try multiple strategies
    strategies = [
        "priority_based",
        "randomized_high_score", 
        "round_robin",
        "scarcity_aware"
    ]
    
    st.write("Trying multiple optimization strategies...")
    strategy_progress = st.progress(0)
    
    for strategy_idx, strategy in enumerate(strategies):
        st.write(f"Trying strategy: {strategy}")
        trackers = reset_trackers()
        
        if strategy == "priority_based":
            # Multiple passes prioritizing students with fewer meetings
            for priority_round in range(student_meetings_target + 1):  # 0, 1, 2, ... target meetings
                students_with_few_meetings = [
                    sid for sid, count in trackers['student_meeting_counts'].items() 
                    if count <= priority_round
                ]
                
                if not students_with_few_meetings:
                    continue
                
                # Get meetings for these students, sorted by score
                round_meetings = meetings_df[
                    meetings_df['student_id'].isin(students_with_few_meetings)
                ].sort_values('score', ascending=False)
                
                assign_meetings(round_meetings, trackers, student_meetings_target, mentor_meetings_max)
        
        elif strategy == "randomized_high_score":
            # Add randomization to break ties among high-scoring matches
            shuffled_meetings = meetings_df.sample(frac=1, random_state=42).sort_values('score', ascending=False)
            assign_meetings(shuffled_meetings, trackers, student_meetings_target, mentor_meetings_max)
        
        elif strategy == "round_robin":
            # Ensure each student gets at least one meeting before anyone gets a second
            for round_num in range(student_meetings_target):
                students_needing_meetings = [
                    sid for sid, count in trackers['student_meeting_counts'].items() 
                    if count == round_num
                ]
                
                # Shuffle students to ensure fairness
                student_list = list(students_needing_meetings)
                random.shuffle(student_list)
                
                for student_id in student_list:
                    student_meetings = meetings_df[
                        meetings_df['student_id'] == student_id
                    ].sort_values('score', ascending=False)
                    
                    # Try to assign one meeting for this student
                    for _, meeting in student_meetings.iterrows():
                        if can_assign_meeting(meeting, trackers, student_meetings_target, mentor_meetings_max):
                            make_assignment(meeting, trackers)
                            break
        
        elif strategy == "scarcity_aware":
            # Prioritize matches with mentors who have fewer available slots
            def calculate_mentor_scarcity():
                return {mid: mentor_meetings_max - count for mid, count in trackers['mentor_meeting_counts'].items()}
            
            remaining_meetings = meetings_df.copy()
            
            while len(remaining_meetings) > 0:
                mentor_scarcity = calculate_mentor_scarcity()
                
                # Add scarcity bonus to scores
                remaining_meetings['adjusted_score'] = remaining_meetings.apply(
                    lambda row: row['score'] + (mentor_scarcity[row['mentor_id']] * 0.1), axis=1
                )
                
                # Sort by adjusted score
                remaining_meetings = remaining_meetings.sort_values('adjusted_score', ascending=False)
                
                assigned_any = False
                for idx, meeting in remaining_meetings.iterrows():
                    if can_assign_meeting(meeting, trackers, student_meetings_target, mentor_meetings_max):
                        make_assignment(meeting, trackers)
                        # Remove this meeting from consideration
                        remaining_meetings = remaining_meetings.drop(idx)
                        assigned_any = True
                        break
                
                if not assigned_any:
                    break
        
        # Evaluate this strategy
        completed_students = sum(1 for count in trackers['student_meeting_counts'].values() if count == student_meetings_target)
        
        st.write(f"Strategy '{strategy}' results: {completed_students}/{len(students_df)} students with 3 meetings")
        
        if completed_students > best_completed_students:
            best_completed_students = completed_students
            best_result = trackers
        
        strategy_progress.progress((strategy_idx + 1) / len(strategies))
    
    # Convert best result to DataFrame
    schedule_df = pd.DataFrame(best_result['assignments'])
    all_matched = all(count == student_meetings_target for count in best_result['student_meeting_counts'].values())
    
    st.success(f"Best result: {best_completed_students}/{len(students_df)} students with {student_meetings_target} meetings ({best_completed_students/len(students_df)*100:.1f}% success rate)")
    
    return schedule_df, all_matched, best_result['student_meeting_counts']

def generate_sample_data():
    """Generate sample data for testing"""
    # Generate 10 students with interests
    students = []
    interests = ["Data Science", "Machine Learning", "Web Development", 
                "Mobile Apps", "Cloud Computing", "Cybersecurity", 
                "UI/UX Design", "Product Management", "Entrepreneurship", 
                "Artificial Intelligence", "Blockchain", "IoT"]
    
    for i in range(1, 11):
        # Randomly select 3-5 interests
        num_interests = np.random.randint(2, 6)
        student_interests = np.random.choice(interests, num_interests, replace=False)
        
        student = {
            'StudentID': f'S{i:03}',
            'Student Name': f'Student {i}',
        }
        
        # Add interests columns
        for j in range(5):  # Max 5 interest columns
            if j < len(student_interests):
                student[f'Interest{j+1}'] = student_interests[j]
            else:
                student[f'Interest{j+1}'] = ''
        
        students.append(student)
    
    # Generate 5 mentors with interests
    mentors = []
    for i in range(1, 6):
        # Randomly select 3-5 interests
        num_interests = np.random.randint(2, 6)
        mentor_interests = np.random.choice(interests, num_interests, replace=False)
        
        mentor = {
            'MentorID': f'M{i:03}',
            'Name': f'Mentor {i}',
        }
        
        # Add interests columns
        for j in range(5):  # Max 5 interest columns
            if j < len(mentor_interests):
                mentor[f'Interest{j+1}'] = mentor_interests[j]
            else:
                mentor[f'Interest{j+1}'] = ''
        
        mentors.append(mentor)
    
    return pd.DataFrame(students), pd.DataFrame(mentors)

def get_excel_download_link(df, filename, sheet_name):
    """Generate a download link for dataframe as Excel file"""
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.close()
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">Download {filename}</a>'
    return href

# Sidebar for data input
st.sidebar.header("Input Data")

data_source = st.sidebar.radio(
    "Select data source",
    ["Upload Files", "Use Sample Data"]
)

if data_source == "Upload Files":
    students_file = st.sidebar.file_uploader("Upload Students CSV", type=["csv"])
    mentors_file = st.sidebar.file_uploader("Upload Mentors CSV", type=["csv"])
    
    if students_file is not None and mentors_file is not None:
        try:
            # Try reading with UTF-8 first
            students_df = pd.read_csv(students_file)
        except UnicodeDecodeError:
            # If UTF-8 fails, try with cp1252 (Windows encoding)
            students_df = pd.read_csv(students_file, encoding='cp1252')
        except Exception as e:
            st.sidebar.error(f"Error reading students file: {str(e)}")
            students_df = None
        
        try:
            # Try reading with UTF-8 first
            mentors_df = pd.read_csv(mentors_file)
        except UnicodeDecodeError:
            # If UTF-8 fails, try with cp1252 (Windows encoding)
            mentors_df = pd.read_csv(mentors_file, encoding='cp1252')
        except Exception as e:
            st.sidebar.error(f"Error reading mentors file: {str(e)}")
            mentors_df = None
        
        # Clean up any completely empty columns
        if students_df is not None:
            students_df = students_df.dropna(axis=1, how='all')
        if mentors_df is not None:
            mentors_df = mentors_df.dropna(axis=1, how='all')
            
        if students_df is not None and mentors_df is not None:
            st.sidebar.success("Files uploaded successfully!")
            st.sidebar.write(f"Students: {len(students_df)}")
            st.sidebar.write(f"Mentors: {len(mentors_df)}")
        else:
            st.sidebar.error("Failed to read one or both files.")
    else:
        st.sidebar.info("Please upload both files to continue.")
        students_df = None
        mentors_df = None
else:
    # Generate sample data
    students_df, mentors_df = generate_sample_data()
    st.sidebar.success("Sample data generated!")
    
    # Show download links for the templates
    st.sidebar.markdown("### Download Sample Data")
    st.sidebar.markdown(get_excel_download_link(students_df, "students_template", "Students"), unsafe_allow_html=True)
    st.sidebar.markdown(get_excel_download_link(mentors_df, "mentors_template", "Mentors"), unsafe_allow_html=True)

# Meeting limits input
st.sidebar.header("Meeting Limits")
student_meetings_target = st.sidebar.number_input(
    "Target meetings per student", 
    min_value=1, 
    max_value=10, 
    value=3,
    help="How many mentors each student should meet with"
)
mentor_meetings_max = st.sidebar.number_input(
    "Maximum meetings per mentor", 
    min_value=1, 
    max_value=20, 
    value=5,
    help="Maximum number of students each mentor can meet with"
)

# Time slots input
st.sidebar.header("Time Slots")
start_time = st.sidebar.time_input("Start time", value=pd.to_datetime('16:30').time())
end_time = st.sidebar.time_input("End time", value=pd.to_datetime('18:30').time())
slot_duration = st.sidebar.number_input("Session duration (minutes)", min_value=5, max_value=30, value=10)
buffer_time = st.sidebar.number_input("Buffer time between sessions (minutes)", min_value=0, max_value=15, value=5)

# Input validation
if start_time >= end_time:
    st.sidebar.error("End time must be after start time.")
else:
    # Generate time slots (with buffer time)
    time_slots = []
    formatted_time_slots = []
    current_time = pd.to_datetime(start_time.strftime('%H:%M'))
    end_datetime = pd.to_datetime(end_time.strftime('%H:%M'))
    
    while current_time < end_datetime:
        # Add the slot start time
        slot_start = current_time.strftime('%H:%M')
        
        # Calculate slot end time (for display purposes)
        slot_end = (current_time + pd.Timedelta(minutes=slot_duration)).strftime('%H:%M')
        
        # Create a formatted time slot string
        formatted_slot = f"{slot_start} - {slot_end}"
        
        time_slots.append(slot_start)
        formatted_time_slots.append(formatted_slot)
        
        # Move to next slot (including buffer time)
        current_time += pd.Timedelta(minutes=slot_duration + buffer_time)

# Capacity check
if students_df is not None and mentors_df is not None:
    required_meetings = len(students_df) * student_meetings_target
    max_mentor_meetings = len(mentors_df) * mentor_meetings_max
    max_time_slot_meetings = len(time_slots) * len(mentors_df)
    
    st.sidebar.header("Capacity Analysis")
    st.sidebar.write(f"**Required meetings:** {required_meetings}")
    st.sidebar.write(f"**Max mentor capacity:** {max_mentor_meetings}")
    st.sidebar.write(f"**Max time slot capacity:** {max_time_slot_meetings}")
    
    if required_meetings > min(max_mentor_meetings, max_time_slot_meetings):
        st.sidebar.error("⚠️ Insufficient capacity! You need more mentors or time slots.")
        if required_meetings > max_mentor_meetings:
            needed_mentors = math.ceil(required_meetings / mentor_meetings_max)
            st.sidebar.write(f"💡 Suggestion: You need at least {needed_mentors} mentors")
        if required_meetings > max_time_slot_meetings:
            needed_slots = math.ceil(required_meetings / len(mentors_df))
            st.sidebar.write(f"💡 Suggestion: You need at least {needed_slots} time slots")
    else:
        st.sidebar.success("✅ Sufficient capacity available")

# Generate matches button
if students_df is not None and mentors_df is not None:
    if st.sidebar.button("Generate Matches"):
        # Display the input data
        with st.expander("Students Data"):
            st.dataframe(students_df)
        
        with st.expander("Mentors Data"):
            st.dataframe(mentors_df)
        
        with st.expander("Time Slots"):
            # Display time slots with duration information
            st.write(f"Session duration: {slot_duration} minutes with {buffer_time} minute buffer")
            st.write(formatted_time_slots)
        
        # Run the matching algorithm
        with st.spinner("Generating optimal matches using advanced algorithm..."):
            # Use the enterprise algorithm with custom meeting limits
            schedule_df, all_matched, student_meeting_counts = generate_matches_enterprise(
                students_df, mentors_df, time_slots, student_meetings_target, mentor_meetings_max
            )
            
            # Set session state
            st.session_state.schedule_generated = True
            st.session_state.all_students_matched = all_matched
            st.session_state.student_meetings_target = student_meetings_target
            
            # Save results to session state
            st.session_state.schedule_df = schedule_df
            st.session_state.student_meeting_counts = student_meeting_counts

# Display results if available
if st.session_state.schedule_generated:
    st.header("Matching Results")
    
    if not st.session_state.all_students_matched:
        target_meetings = getattr(st.session_state, 'student_meetings_target', 3)
        st.warning(f"⚠️ Not all students received {target_meetings} meetings. You may need more mentors or time slots.")
        
        # Show student meeting counts
        meeting_counts_df = pd.DataFrame({
            'StudentID': list(st.session_state.student_meeting_counts.keys()),
            'Meetings Assigned': list(st.session_state.student_meeting_counts.values())
        })
        meeting_counts_df = meeting_counts_df.sort_values('Meetings Assigned')
        
        st.subheader("Students with Incomplete Schedules")
        incomplete_students = meeting_counts_df[meeting_counts_df['Meetings Assigned'] < target_meetings]
        st.dataframe(incomplete_students)
    else:
        target_meetings = getattr(st.session_state, 'student_meetings_target', 3)
        st.success(f"🎉 All students successfully matched with {target_meetings} mentors!")
    
    # Calculate and display match quality statistics
    if len(st.session_state.schedule_df) > 0:
        avg_score = st.session_state.schedule_df['score'].mean()
        min_score = st.session_state.schedule_df['score'].min()
        max_score = st.session_state.schedule_df['score'].max()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Meetings", len(st.session_state.schedule_df))
        with col2:
            st.metric("Average Match Score", f"{avg_score:.1%}")
        with col3:
            st.metric("Min Match Score", f"{min_score:.1%}")
        with col4:
            st.metric("Max Match Score", f"{max_score:.1%}")
    
    # Display the schedule
    st.subheader("Complete Schedule")
    
    # Sort by time slot, then student
    sorted_schedule = st.session_state.schedule_df.sort_values(['time_slot', 'student_name'])
    
    # Format for display
    display_schedule = sorted_schedule[['time_slot', 'student_name', 'mentor_name', 'score']].copy()
    display_schedule = display_schedule.rename(columns={
        'time_slot': 'Time',
        'student_name': 'Student',
        'mentor_name': 'Mentor',
        'score': 'Match Score'
    })
    
    # Format score as percentage
    display_schedule['Match Score'] = display_schedule['Match Score'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(display_schedule, use_container_width=True)
    
    # Student view
    st.subheader("Student Schedules")
    student_schedule = sorted_schedule.pivot_table(
        index='student_name',
        columns='time_slot',
        values='mentor_name',
        aggfunc='first'
    ).reset_index()
    
    st.dataframe(student_schedule, use_container_width=True)
    
    # Mentor view
    st.subheader("Mentor Schedules")
    mentor_schedule = sorted_schedule.pivot_table(
        index='mentor_name',
        columns='time_slot',
        values='student_name',
        aggfunc='first'
    ).reset_index()
    
    st.dataframe(mentor_schedule, use_container_width=True)
    
    # Download links
    st.subheader("Download Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(get_excel_download_link(display_schedule, "complete_schedule", "Schedule"), unsafe_allow_html=True)
    with col2:
        st.markdown(get_excel_download_link(student_schedule, "student_schedules", "Student Schedules"), unsafe_allow_html=True)
    with col3:
        st.markdown(get_excel_download_link(mentor_schedule, "mentor_schedules", "Mentor Schedules"), unsafe_allow_html=True)

# Instructions at the bottom
with st.expander("Instructions"):
    st.write("""
    ### How to use this app:
    
    1. **Input data**: Upload CSV files with student and mentor information, or use the sample data.
    2. **Set time slots**: Configure your event timing (default: 4:30 PM to 6:30 PM with 10-minute sessions).
    3. **Check capacity**: The app will verify if you have enough mentors and time slots.
    4. **Generate matches**: Click the button to create optimal matches using our advanced algorithm.
    5. **Review results**: Check the generated schedules and download as Excel files.
    
    ### Required CSV format:
    
    **Students CSV**:
    - `StudentID`: Unique identifier for each student
    - `Student Name`: Student's name
    - `Interest1`, `Interest2`, etc.: Columns containing student interests
    
    **Mentors CSV**:
    - `MentorID`: Unique identifier for each mentor
    - `Name`: Mentor's name
    - `Interest1`, `Interest2`, etc.: Columns containing mentor interests
    
    ### Algorithm Features:
    
    - **Interest-based matching**: Students are paired with mentors based on shared interests
    - **Multiple optimization strategies**: Tries different approaches to find the best solution
    - **Fairness guarantee**: Prioritizes students with fewer meetings to ensure everyone gets 3 sessions
    - **Capacity analysis**: Checks if your configuration is mathematically feasible
    - **Quality metrics**: Shows match scores and success rates
    
    You can download the sample data to see the expected format.
    """)
