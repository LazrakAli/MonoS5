.data 

octet:	.byte	0xFF	#0(0x1001)
	.align	2	#1(0x1001)
saut:	.asciiz "\n"	#4(0x1001)
	.align	2	#10(0x1001)
.text 
#affichage octets signé
	lui	$8, 0x1001
	lb	$9, 0($8)
	ori	$4, $9, 0
	ori	$2, $0, 1
	syscall
	
#affichage du saut de ligne 
	lui	$8,0x1001
	lw	$10, 4($8)
	ori	$4, $10, 0
	ori 	$2, $0, 11
	syscall

#affichage octets non signé
	lui	$8, 0x1001
	lbu	$10, 0($8)
	ori	$4, $10, 0
	ori	$2, $0, 1
	syscall

#exit() 
	ori 	$2, $0, 10
	syscall