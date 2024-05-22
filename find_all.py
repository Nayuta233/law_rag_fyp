import re
str2 = "aabab"
a = re.findall('a.*?b',str2)	#结果：['aab', 'ab']
b = re.findall('a.+?b',str2)	#结果：['aab']