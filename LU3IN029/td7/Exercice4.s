.data

tab1:	.word	23, 4, 5, -1
tab2:	.word	2, 345, 56, 23, 45, -1

.text

main:
# prologue
	addiu $29, $29, -4
	
 # nb_elem(tab1);
	lui	$8,0x1001
	ori	$4,$8,0
	jal	nb_eleme

# printf("%d", nb_elem(tab1))
	ori	$4,$2,0
	ori	$2,$0,1
	syscall

# printf("\n"); /* affichage du caractere retour ` a la ligne ` */
	ori	$4,$0,0X0A
	ori	$2,$0,11
	syscall

#  nb_elem(tab2)
	lui	$4,0x1001
	ori	$4,$4,16
	jal	nb_eleme

# printf("%d", nb_elem(tab2))
	ori	$4,$2,0
	ori	$2,$0,1
	syscall
#epiologue
	addiu $29, $29, -4

# exit	
	ori	$2,$0,10
	syscall

	


nb_eleme:
# prologue
	addiu	$29,$29,-12
	sw	$31,8($29)
	addiu	$9,$0,-1
	ori	$2,$0,0
	ori	$10,$0,0
while:
	lw	$8,0($4)
	bltz	$8, endwhile
	addiu	$10,$10,1
	addiu	$4,$4,4
	j	while

endwhile:
	ori	$2,$10,0
#prologue
	lw	$31,8($29)
	addiu	$29,$29,12
	jr	$31
	