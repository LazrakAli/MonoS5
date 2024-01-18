.data 

v1:	.word	-1	#0x1001
v2:	.word	0xFF	#4(0x1001)
saut:	.asciiz "\n"	#8(0x1001)
	.align	2	#10(0x1001)


.text
#ajout 1 a v1 et a v2
	lui $8,0x1001
	lw  $9,0($8)
	lw  $10,4($8)
	
	addi $9, $9, 1
	addi $10, $10, 1
	
	sw   $10, 4($8)
	sw   $9, 0($8)
#affichage du mot 1
	lui	$8,0x1001	
	lw	$8,0($8)
	ori	$4, $8 ,0
	ori	$2, $0, 1
	syscall
	
#affichage du saut de ligne 
	lui	$8,0x1001
	lw	$10, 8($8)
	ori	$4, $10, 0
	ori 	$2, $0, 11
	syscall

#affichage du mot2
	lui	$8,0x1001
	lw	$9, 4($8)
	ori	$4, $9, 0
	ori 	$2, $0, 1
	syscall

	
#exit() 
	ori 	$2, $0, 10
	syscall