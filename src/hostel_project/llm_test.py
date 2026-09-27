from hostel_project.small_llm import generate_response

prompt='''

You are a hostel assistant for NIT Jalandhar.

Answer this question briefly:

What is a hostel?


'''
answer=generate_response(prompt)
print('\nAnswer:')
print(answer)