import random
from datetime import datetime, timedelta
from django.contrib.auth.models import User

class Persons:

    def __init__(self):

        self.first_names = ["John", "Jane", "Alex", "Chris", "Taylor", "Morgan", "Jordan", "Cameron", "Casey", "Sam",
                    "Olivia", "Liam", "Emma", "Noah", "Ava", "Sophia", "Mason", "Isabella", "James", "Mia",
                    "Charlotte", "Amelia", "Evelyn", "Harper", "Benjamin", "Elijah", "Lucas", "Henry", "Oliver", "Jack"]

        self.middle_names = ["Lee", "Ray", "Ann", "James", "Marie", "Lynn", "Drew", "Blake", "Reese", "Sky",
                        "Grace", "Hope", "Jay", "Ryan", "Jude", "Elle", "Dane", "Kai", "Rose", "Brooke"]

        self.last_names = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor",
                    "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Garcia", "Martinez", "Robinson",
                    "Clark", "Rodriguez", "Lewis", "Lee", "Walker", "Hall", "Allen", "Young", "King", "Wright"]
        
    def generate_first_name(self):
        return random.choice(self.first_names)
    
    def generate_middle_name(self):
        return random.choice(self.middle_names)
    
    def generate_last_name(self):
        return random.choice(self.last_names)
    
    def generate_birth_date(self):

        start_date = datetime(1960, 1, 1)
        end_date = datetime(2020, 12, 31)
        delta = end_date - start_date
        random_days = random.randint(0, delta.days)
        
        return start_date + timedelta(days=random_days)

class UserGenerator:

    def __init__(self):
        self.default_password = 'W3lc0m3!1234'

    def generate_email(self, name, surname):
        
        domains = [
            '@webmailer.com', '@gmail.com', '@yahoo.com',
            '@hotmail.com'
        ]

        return str(name).lower() + str(surname).lower() + random.choice(domains)
    
    def generate_random_number(self):

        numbers = []
        for i in range(100):
            numbers.append(i)
        
        return str(random.choice(numbers))

    def generate_user(self):
        Person = Persons()
        first_name = Person.generate_first_name()
        last_name = Person.generate_last_name()
        email = self.generate_email(first_name, last_name)

        data = {
            "username": first_name.lower() + last_name.lower() + self.generate_random_number(),
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "password": self.default_password,  # Assumes it's a plain string to be hashed
        }

        # Create the user
        user = User.objects.create_user(**data)

        # Set is_active to False if needed
        user.is_active = False
        user.save()

        return user
        

class PatientGenerator:

    def __init__(self):
        self.patients = self.generate_random_names()

    def generate_random_names(self):

        Person = Persons()

        # Expanded sample data for names
        # Creating lists of patients
        patients = []

        for _ in range(100):

            patient = {
                "first_name": random.choice(Person.generate_first_name()),
                "middle_name": random.choice(Person.generate_middle_name()) if random.random() > 0.2 else None,  # 20% chance to be None
                "last_name": random.choice(Person.generate_last_name()),
                "birth": Person.generate_birth_date().strftime("%Y-%m-%d")
            }
            
            patients.append(patient)

        return patients


