.data
.text
# scanf
	ori	$2,$0,5
	syscall

	addiu	$4, $2, 10
	
	ori	$2, $0 ,1
	syscall
	
	ori	$2, $0 , 10 #exit
	syscall
	 