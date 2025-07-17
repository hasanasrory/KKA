suhu = int(input('Masukkan suhu: '))

if suhu <= 20:
	print('Dingin')
elif 20 <= suhu and suhu <= 30:
	print('Sejuk')
else:
	print('Panas')
