.data

#123456\0
tab:	.byte	0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x00 #0x10010000->7 octes

.text
	lui	$8,0x1001	#$8->tab
	ori	$4,$8,0
	ori	$2,$0,4
	syscall
	
#saut de ligne
	ori	$4,$0,'\n'
	ori	$2,$0,11
	syscall
	
	
#3eme caractere->$16, Affichez sa valeur en decimal
	lui	$8,0x1001
	lb	$16, 2($8)
	ori	$4,$16,0
	ori	$2,$0,1
	syscall	

	
#exit()
	ori	$2,$0,10
	syscall