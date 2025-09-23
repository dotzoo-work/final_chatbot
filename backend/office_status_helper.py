"""
Office Status Helper - Centralized logic for checking office availability
"""

from datetime import datetime, timedelta
from typing import Dict
import pytz

def get_next_open_day() -> str:
    """Get the next open day from today"""
    pacific_tz = pytz.timezone('America/Los_Angeles')
    now = datetime.now(pacific_tz)
    open_days = {'Monday', 'Tuesday', 'Thursday'}
    
    # Check next 7 days
    for i in range(1, 8):
        next_date = now + timedelta(days=i)
        next_day = next_date.strftime('%A')
        if next_day in open_days:
            return next_day
    
    return 'Monday'  # fallback

def get_dynamic_followup_question() -> str:
    """Generate dynamic follow-up questions for scheduling responses"""
    import random
    
    followup_questions = [
        "What type of dental concern would you like to address during your visit? 🦷",
        "Is this for a routine cleaning or do you have a specific dental concern? 🦷",
        "Are you experiencing any dental pain or discomfort? 🦷",
        "Would you like to schedule a consultation or cleaning appointment? 🦷",
        "Do you have any specific dental issues you'd like Dr. Tomar to examine? 🦷",
        "Is this for preventive care or do you need treatment for a dental problem? 🦷",
        "What brings you to our dental office today? 🦷",
        "Are you looking for a routine checkup or addressing a dental concern? 🦷"
    ]
    
    return random.choice(followup_questions)

def check_office_status(day: str) -> Dict[str, any]:
    """
    3-level check for office status:
    1. Check if it's an open day (Monday/Tuesday/Thursday)
    2. If open day, check current time
    3. Return appropriate status message
    """
    
    open_days = {'Monday', 'Tuesday', 'Thursday'}
    pacific_tz = pytz.timezone('America/Los_Angeles')
    now = datetime.now(pacific_tz)
    current_day = now.strftime('%A')
    
    # Level 1: Check if it's an open day
    if day not in open_days:
        # Level 3: Not an open day
        next_open = get_next_open_day()
        return {
            'is_open': False,
            'hours': 'Closed',
            'day': day,
            'status_message': f'Closed, next open {next_open} 7 AM to 6 PM.'
        }
    
    # Level 2: It's an open day, check time (only for today)
    if day == current_day:
        current_hour = now.hour
        
        if current_hour < 7:
            # Before business hours
            return {
                'is_open': False,
                'hours': 'Currently closed',
                'day': day,
                'status_message': 'Currently closed, today we open at 7 AM to 6 PM.'
            }
        elif 7 <= current_hour < 18:
            # Within business hours
            return {
                'is_open': True,
                'hours': 'Open until 6 PM',
                'day': day,
                'status_message': 'Open until 6 PM.'
            }
        else:
            # After business hours
            next_open = get_next_open_day()
            return {
                'is_open': False,
                'hours': 'Currently closed',
                'day': day,
                'status_message': f'Currently closed, next open {next_open} 7 AM to 6 PM.'
            }
    else:
        # For future open days
        return {
            'is_open': True,
            'hours': '7:00 AM - 6:00 PM',
            'day': day,
            'status_message': f'{day} is open from 7 AM to 6 PM'
        }