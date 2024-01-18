.data
ch:	.asciiz	""
.text

	lui	$8,0x1001
#tab[0]

# Compteur
	xor	$15,$15,$15
while:
#condition: len(tab)=n => tab[n]=\0
	lb	$9,0($8)
	beq	$9,$0,endwhile
	addi	$15,$15,1
	addiu	$8,$8,1
	j while
	
endwhile:
#affichage
	ori	$4,$15,0
	ori	$2,$0,1
	syscall
#exit	
	ori	$2,$0,10
	syscall
	